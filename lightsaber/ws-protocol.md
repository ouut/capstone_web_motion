# WebSocket Lightsaber Data Protocol

## Overview

Each frame, the client sends one binary WebSocket frame encoded with the gateway protocol. The payload contains two lightsabers (left + right), each with a hilt position and blade tip position in 3D camera space.

## Gateway Wire Format

Every binary WebSocket message uses the gateway protocol header:

```
Byte 0     : version         = 0x02
Byte 1     : pktType         = 0x01  (PKT_RAW_MOTION)
Byte 2     : tgtType         = 0x01  (TGT_BROADCAST)
Byte 3     : reserved        = 0x00
Bytes 4-9  : roomId          (6 bytes ASCII, zero-padded)
Bytes 10-17: userId          (8 bytes ASCII, zero-padded)
Bytes 18-21: seq             (uint32 big-endian — frame counter)
Bytes 22-23: payloadLength   (uint16 big-endian — always 48)
Bytes 24-71: payload         (48 bytes — the lightsaber data)
```

Total frame size: **72 bytes** (24 header + 48 payload).

See `gateway_protocol.js` for reference encode/decode implementation.

## Payload Layout (48 bytes)

The payload contains **12 float32 values** (little-endian, 4 bytes each):

```
Offset  Size   Field
─────────────────────────────────────────
  0      4     Left saber hilt X  (float32 LE)
  4      4     Left saber hilt Y
  8      4     Left saber hilt Z
 12      4     Left saber tip X
 16      4     Left saber tip Y
 20      4     Left saber tip Z
 24      4     Right saber hilt X
 28      4     Right saber hilt Y
 32      4     Right saber hilt Z
 36      4     Right saber tip X
 40      4     Right saber tip Y
 44      4     Right saber tip Z
```

Coordinate system: **right-handed, Y-up**

- X → right
- Y → up
- Z → forward (away from camera)

## Data Meaning

### Hands Mode (✋ Hands)

| Saber  | Hilt                | Tip                      | Constraint |
|--------|---------------------|--------------------------|------------|
| Left   | Wrist 3D position   | Extends toward Index_TIP | ~18 cm     |
| Right  | Wrist 3D position   | Extends toward Index_TIP | ~18 cm     |

### Body Mode (🧍 Body)

| Saber  | Hilt                | Tip                  | Constraint |
|--------|---------------------|----------------------|------------|
| Left   | Elbow 3D position   | Extends toward Wrist | ~29 cm     |
| Right  | Elbow 3D position   | Extends toward Wrist | ~29 cm     |

### Missing Data

When no hand is detected (e.g., only one hand in frame), the corresponding saber sends **all zeros** (hilt and tip at origin). Check `hilt[0]==0 && hilt[1]==0 && hilt[2]==0` to detect missing sabers.

## Receiver Parsing (Pseudocode)

### JavaScript (Web)

```js
// In browser — receive via WebSocket
ws.binaryType = 'arraybuffer';
ws.onmessage = (event) => {
  const buf = new Uint8Array(event.data);

  // Decode gateway header (or use gateway_protocol.js)
  const decoded = gatewayProtocol.decode(buf);

  // Parse payload: 12 × float32 LE
  const dv = new DataView(decoded.payload.buffer);
  const sabers = {
    left: {
      hilt: [dv.getFloat32(0, true),  dv.getFloat32(4, true),  dv.getFloat32(8, true)],
      tip:  [dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true)],
    },
    right: {
      hilt: [dv.getFloat32(24, true), dv.getFloat32(28, true), dv.getFloat32(32, true)],
      tip:  [dv.getFloat32(36, true), dv.getFloat32(40, true), dv.getFloat32(44, true)],
    },
  };

  // Filter empty sabers
  if (isActive(sabers.left))  updateLeftSaber(sabers.left);
  if (isActive(sabers.right)) updateRightSaber(sabers.right);
};

function isActive(saber) {
  const [x, y, z] = saber.hilt;
  return !(x === 0 && y === 0 && z === 0);
}
```

### Unity (C#)

```csharp
public struct SaberFrame {
    public Vector3 leftHilt, leftTip;
    public Vector3 rightHilt, rightTip;
}

SaberFrame ParseSaberPayload(byte[] payload, int offset = 0) {
    var f = new SaberFrame();
    f.leftHilt  = new Vector3(BitConverter.ToSingle(payload, offset),
                              BitConverter.ToSingle(payload, offset + 4),
                              BitConverter.ToSingle(payload, offset + 8));
    f.leftTip   = new Vector3(BitConverter.ToSingle(payload, offset + 12),
                              BitConverter.ToSingle(payload, offset + 16),
                              BitConverter.ToSingle(payload, offset + 20));
    f.rightHilt = new Vector3(BitConverter.ToSingle(payload, offset + 24),
                              BitConverter.ToSingle(payload, offset + 28),
                              BitConverter.ToSingle(payload, offset + 32));
    f.rightTip  = new Vector3(BitConverter.ToSingle(payload, offset + 36),
                              BitConverter.ToSingle(payload, offset + 40),
                              BitConverter.ToSingle(payload, offset + 44));
    return f;
}
```

### Unreal (Blueprint / C++)

```
// C++ — after decoding gateway header, read payload:
float LeftHiltX, LeftHiltY, LeftHiltZ;
float LeftTipX,  LeftTipY,  LeftTipZ;
float RightHiltX, RightHiltY, RightHiltZ;
float RightTipX,  RightTipY,  RightTipZ;

// Read as FMemory::Memcpy from payload buffer at offset 0..47
// Use PLATFORM_LITTLE_ENDIAN to verify byte order
```

## Coordinate Usage in Game Engines

The coordinates are in **camera space**:
- Origin is at the camera
- X right, Y up, Z forward

To place in world space, apply the camera's world transform:

```csharp
// Unity example
Vector3 worldHilt = cameraTransform.TransformPoint(saber.hilt);
```

The **saber direction vector** is:
```js
const direction = normalize([
  saber.tip[0] - saber.hilt[0],
  saber.tip[1] - saber.hilt[1],
  saber.tip[2] - saber.hilt[2],
]);
```

The **saber rotation quaternion** (if needed) can be computed by finding the rotation that maps the saber's default forward axis to this direction vector.

## Limitations

1. **Depth (Z) is reconstructed, not measured** — Monocular camera cannot provide true depth. The tracker uses IK constraints and assumed arm length. Z values have higher uncertainty than X/Y.
2. **Requires calibration** — The `shoulderPos`, `armLength`, and `handLength` parameters assume average human proportions. Adjust for your setup.
3. **No occlusion handling** — If a hand is hidden behind the body, data goes to zero.
4. **One person only** — The Body mode assumes one detected person.
