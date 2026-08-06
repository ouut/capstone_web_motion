/**
 * 光剑追踪器 - 从身体关键点重建 3D 光剑姿态
 * 支持两种模式：
 * 1. 手腕+食指 -> 光剑（较短的约束长度）
 * 2. 手腕+肘部 -> 身体光剑（较长的手臂约束长度）
 */

class LightsaberTracker {
  /**
   * @param {Object} config - 配置参数
   * @param {number[]} config.shoulderPos - 肩膀在 3D 空间的位置 [x, y, z]
   * @param {number} config.armLength - 手臂总长度（米），默认 0.65
   * @param {number} config.handLength - 手腕到食指的长度（米），默认 0.18
   * @param {number[]} config.cameraForward - 相机前方向量 [x, y, z]，默认 [0, 0, 1]
   * @param {number[]} config.imageSize - 图像尺寸 [width, height]
   * @param {number} config.fov - 相机视场角（度），默认 60
   * @param {number} config.smoothing - 平滑系数 0-1，默认 0.7
   */
  constructor(config = {}) {
    this.shoulder = config.shoulderPos || [0, -0.15, -0.4];
    this.armLength = config.armLength || 0.65;
    this.handLength = config.handLength || 0.18;
    this.forward = this.normalize(config.cameraForward || [0, 0, 1]);
    this.imageSize = config.imageSize || [640, 480];
    this.fov = config.fov || 60;
    this.smoothing = config.smoothing || 0.7;
    
    // 平滑状态存储
    this.state = {
      left: { wrist3D: null, bladeDir: null },
      right: { wrist3D: null, bladeDir: null }
    };
  }

  /**
   * 主入口：处理左右手的光剑追踪
   * @param {Object} landmarks - MediaPipe 风格的归一化关键点 (0-1)
   * @param {Object} landmarks.leftWrist - 左手腕 [x, y]
   * @param {Object} landmarks.leftIndex - 左手食指指尖 [x, y]
   * @param {Object} landmarks.rightWrist - 右手腕 [x, y]
   * @param {Object} landmarks.rightIndex - 右手食指指尖 [x, y]
   * @returns {Object} 左右光剑的姿态信息
   */
  processHandLightsabers(landmarks) {
    const left = this.reconstructLightsaber(
      landmarks.leftWrist,
      landmarks.leftIndex,
      'left',
      this.handLength  // 手指光剑用较短约束
    );
    
    const right = this.reconstructLightsaber(
      landmarks.rightWrist,
      landmarks.rightIndex,
      'right',
      this.handLength
    );
    
    return { left, right };
  }

  /**
   * 处理身体光剑（用手腕和肘部）
   * @param {Object} landmarks - 关键点
   * @param {Object} landmarks.leftWrist - 左手腕 [x, y]
   * @param {Object} landmarks.leftElbow - 左肘部 [x, y]
   * @param {Object} landmarks.rightWrist - 右手腕 [x, y]
   * @param {Object} landmarks.rightElbow - 右肘部 [x, y]
   * @returns {Object} 左右身体光剑的姿态
   */
  processBodyLightsabers(landmarks) {
    const forearmLength = this.armLength * 0.45;  // 前臂长度约束
    
    const left = this.reconstructLightsaber(
      landmarks.leftElbow,   // 起点改为肘部
      landmarks.leftWrist,   // 终点改为手腕
      'left',
      forearmLength
    );
    
    const right = this.reconstructLightsaber(
      landmarks.rightElbow,
      landmarks.rightWrist,
      'right',
      forearmLength
    );
    
    return { left, right };
  }

  /**
   * 核心重建函数：从两个 2D 点重建光剑的 3D 姿态
   * @param {number[]} pointA - 起点（手腕/肘部）的归一化坐标 [x, y] (0-1)
   * @param {number[]} pointB - 终点（食指/手腕）的归一化坐标 [x, y] (0-1)
   * @param {string} side - 'left' 或 'right'
   * @param {number} constraintLength - 约束长度（米）
   * @returns {Object} { position: [x,y,z], rotation: [x,y,z,w], direction: [x,y,z] }
   */
  reconstructLightsaber(pointA, pointB, side, constraintLength) {
    // 1. 转换为像素坐标
    const [w, h] = this.imageSize;
    const aPx = [pointA[0] * w, pointA[1] * h];
    const bPx = [pointB[0] * w, pointB[1] * h];
    
    // 2. 像素坐标 -> 射线方向
    const rayA = this.pixelToRay(aPx);
    const rayB = this.pixelToRay(bPx);
    
    // 3. 求解点 A 的 3D 位置（带约束）
    const a3D = this.solveConstrained3D(rayA, constraintLength);
    
    // 4. 计算光剑方向
    const bladeDir = this.computeBladeDirection(a3D, rayA, rayB, aPx, bPx);
    
    // 5. 施加"剑柄比剑头更靠近身体"约束
    const constrainedDir = this.enforceBladeConstraint(bladeDir);
    
    // 6. 平滑处理
    const smoothed = this.smooth(side, a3D, constrainedDir);
    
    // 7. 转换为旋转四元数
    const rotation = this.directionToQuaternion(smoothed.bladeDir);
    
    return {
      position: smoothed.wrist3D,
      rotation: rotation,
      direction: smoothed.bladeDir
    };
  }

  /**
   * 将 2D 像素坐标转换为 3D 射线方向
   */
  pixelToRay(pixel) {
    const [w, h] = this.imageSize;
    const focal = w / (2 * Math.tan(this.degToRad(this.fov / 2)));
    
    // 归一化设备坐标
    const ndcX = (pixel[0] - w / 2) / focal;
    const ndcY = -(pixel[1] - h / 2) / focal;  // Y 轴反转
    
    const ray = [ndcX, ndcY, 1.0];
    return this.normalize(ray);
  }

  /**
   * 求解点 A 在 3D 空间中的位置
   * 约束：|shoulder + t * ray| = constraintLength
   */
  solveConstrained3D(ray, constraintLength) {
    const s = this.shoulder;
    const v = ray;
    
    // 求解二次方程：|s + t*v|² = L²
    const a = this.dot(v, v);
    const b = 2 * this.dot(s, v);
    const c = this.dot(s, s) - constraintLength * constraintLength;
    
    const discriminant = b * b - 4 * a * c;
    
    let t;
    if (discriminant <= 0) {
      // 无解，取射线上的最近点
      t = -b / (2 * a);
    } else {
      const sqrtDisc = Math.sqrt(discriminant);
      const t1 = (-b + sqrtDisc) / (2 * a);
      const t2 = (-b - sqrtDisc) / (2 * a);
      
      // 选择两个解中使得剑指向外的那个
      // 优先选择较大的 t（较远的解通常对应手臂伸直，更稳定）
      t = Math.max(t1, t2);
      
      // 确保 t 不为负
      t = Math.max(t, 0.1);
    }
    
    return [
      s[0] + t * v[0],
      s[1] + t * v[1],
      s[2] + t * v[2]
    ];
  }

  /**
   * 计算光剑方向（从剑柄到剑尖）
   */
  computeBladeDirection(a3D, rayA, rayB, aPx, bPx) {
    // 方法：利用 2D 偏移 + 射线差异来估算 3D 方向
    
    // 在相机坐标系中构建局部坐标系
    const zAxis = rayA;  // 深度方向
    const worldY = [0, 1, 0];
    const xAxis = this.cross(worldY, zAxis);
    this.normalizeInPlace(xAxis);
    const yAxis = this.cross(zAxis, xAxis);
    
    // 2D 偏移量（归一化）
    const [w, h] = this.imageSize;
    const offset2D = [
      (bPx[0] - aPx[0]) / w,
      -(bPx[1] - aPx[1]) / h  // Y 轴反转
    ];
    
    // 组合 3D 方向：2D偏移在XY平面 + 前向分量
    const bladeDir = [
      offset2D[0] * xAxis[0] + offset2D[1] * yAxis[0] + 0.3 * zAxis[0],
      offset2D[0] * xAxis[1] + offset2D[1] * yAxis[1] + 0.3 * zAxis[1],
      offset2D[0] * xAxis[2] + offset2D[1] * yAxis[2] + 0.3 * zAxis[2]
    ];
    
    return this.normalize(bladeDir);
  }

  /**
   * 强制约束：剑柄必须比剑头更靠近身体
   * 即光剑方向在前向上的投影必须为正
   */
  enforceBladeConstraint(bladeDir) {
    const forwardness = this.dot(bladeDir, this.forward);
    
    // 如果光剑指向身体（负值），翻转方向
    if (forwardness < 0) {
      return [-bladeDir[0], -bladeDir[1], -bladeDir[2]];
    }
    
    return bladeDir;
  }

  /**
   * 时间平滑滤波
   */
  smooth(side, wrist3D, bladeDir) {
    const prev = this.state[side];
    
    if (prev.wrist3D === null) {
      // 首次初始化
      prev.wrist3D = wrist3D;
      prev.bladeDir = bladeDir;
      return { wrist3D, bladeDir };
    }
    
    const alpha = this.smoothing;
    
    // 位置平滑
    const smoothWrist = [
      alpha * prev.wrist3D[0] + (1 - alpha) * wrist3D[0],
      alpha * prev.wrist3D[1] + (1 - alpha) * wrist3D[1],
      alpha * prev.wrist3D[2] + (1 - alpha) * wrist3D[2]
    ];
    
    // 方向平滑（球面线性插值）
    let smoothDir;
    const dot = this.clamp(this.dot(bladeDir, prev.bladeDir), -1, 1);
    const angle = Math.acos(dot);
    
    if (angle > 0.001) {
      const sinAngle = Math.sin(angle);
      const a = Math.sin(alpha * angle) / sinAngle;
      const b = Math.sin((1 - alpha) * angle) / sinAngle;
      smoothDir = [
        a * prev.bladeDir[0] + b * bladeDir[0],
        a * prev.bladeDir[1] + b * bladeDir[1],
        a * prev.bladeDir[2] + b * bladeDir[2]
      ];
      this.normalizeInPlace(smoothDir);
    } else {
      smoothDir = bladeDir;
    }
    
    // 再次强制约束
    smoothDir = this.enforceBladeConstraint(smoothDir);
    
    // 更新状态
    prev.wrist3D = smoothWrist;
    prev.bladeDir = smoothDir;
    
    return { wrist3D: smoothWrist, bladeDir: smoothDir };
  }

  /**
   * 方向向量 -> 旋转四元数
   * 假设光剑模型的默认朝向是 Z 轴（前向）
   */
  directionToQuaternion(direction) {
    const forward = this.normalize(direction);
    const worldUp = [0, 1, 0];
    
    // 如果方向接近世界Y轴，改用其他up向量
    let up;
    if (Math.abs(this.dot(forward, worldUp)) > 0.999) {
      up = [1, 0, 0];
    } else {
      up = worldUp;
    }
    
    // 构建旋转矩阵的基向量
    const right = this.cross(up, forward);
    this.normalizeInPlace(right);
    const correctedUp = this.cross(forward, right);
    
    // 旋转矩阵转四元数
    const m00 = right[0], m01 = correctedUp[0], m02 = forward[0];
    const m10 = right[1], m11 = correctedUp[1], m12 = forward[1];
    const m20 = right[2], m21 = correctedUp[2], m22 = forward[2];
    
    const trace = m00 + m11 + m22;
    let x, y, z, w;
    
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      w = 0.25 / s;
      x = (m21 - m12) * s;
      y = (m02 - m20) * s;
      z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
      const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
      w = (m21 - m12) / s;
      x = 0.25 * s;
      y = (m01 + m10) / s;
      z = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
      w = (m02 - m20) / s;
      x = (m01 + m10) / s;
      y = 0.25 * s;
      z = (m12 + m21) / s;
    } else {
      const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
      w = (m10 - m01) / s;
      x = (m02 + m20) / s;
      y = (m12 + m21) / s;
      z = 0.25 * s;
    }
    
    return [x, y, z, w];
  }

  // ========== 工具函数 ==========
  
  normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-8) return [0, 0, 1];
    return [v[0] / len, v[1] / len, v[2] / len];
  }

  normalizeInPlace(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-8) {
      v[0] /= len;
      v[1] /= len;
      v[2] /= len;
    }
  }

  dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }

  clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  degToRad(degrees) {
    return degrees * Math.PI / 180;
  }
}

// ========== 导出 (Node / AMD / Browser) ==========
if (typeof module !== 'undefined' && module.exports) {
  module.exports = LightsaberTracker;
} else if (typeof define === 'function' && define.amd) {
  define(function () { return LightsaberTracker; });
} else {
  // 浏览器全局
  if (typeof window !== 'undefined') window.LightsaberTracker = LightsaberTracker;
  if (typeof globalThis !== 'undefined') globalThis.LightsaberTracker = LightsaberTracker;
}