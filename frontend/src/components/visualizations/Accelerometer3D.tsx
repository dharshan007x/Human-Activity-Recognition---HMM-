import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw } from 'lucide-react';
import { Button } from '../common/Button';

interface Accelerometer3DProps {
  ax: number;
  ay: number;
  az: number;
  activityName?: string;
  height?: number;
}

export const Accelerometer3D: React.FC<Accelerometer3DProps> = ({
  ax,
  ay,
  az,
  activityName = 'Walking',
  height = 360,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const arrowRef = useRef<THREE.ArrowHelper | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animFrameRef = useRef<number | null>(null);

  const magnitude = Math.sqrt(ax * ax + ay * ay + az * az);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const h = height;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#F3EADB'); // Warm paper tone

    // 2. Camera setup
    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.set(4.5, 3.8, 5.0);
    cameraRef.current = camera;

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 15;
    controls.minDistance = 2;
    controls.target.set(0, 0.2, 0);
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xfff8ee, 0.9);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(5, 8, 4);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const fillLight = new THREE.PointLight(0xc4603a, 0.4, 10);
    fillLight.position.set(-4, 3, -3);
    scene.add(fillLight);

    // 6. Ground Grid
    const gridHelper = new THREE.GridHelper(8, 16, 0xcbb99f, 0xe2d3bc);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // 7. Physical IMU Board (PCB Representation)
    const pcbGroup = new THREE.Group();

    // PCB substrate
    const pcbGeo = new THREE.BoxGeometry(2.4, 0.12, 1.8);
    const pcbMat = new THREE.MeshStandardMaterial({
      color: 0x2b382d, // Dark PCB green
      roughness: 0.4,
      metalness: 0.1,
    });
    const pcbMesh = new THREE.Mesh(pcbGeo, pcbMat);
    pcbMesh.position.y = 0.06;
    pcbMesh.receiveShadow = true;
    pcbMesh.castShadow = true;
    pcbGroup.add(pcbMesh);

    // IMU Chip (IC package)
    const chipGeo = new THREE.BoxGeometry(0.8, 0.14, 0.8);
    const chipMat = new THREE.MeshStandardMaterial({
      color: 0x1f1a17, // Matte black IC
      roughness: 0.2,
      metalness: 0.3,
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);
    chipMesh.position.y = 0.18;
    chipMesh.castShadow = true;
    pcbGroup.add(chipMesh);

    // Chip pin 1 marker (gold dot)
    const pinDotGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.02, 12);
    const goldMat = new THREE.MeshStandardMaterial({
      color: 0xd4af37,
      metalness: 0.8,
      roughness: 0.3,
    });
    const pinDot = new THREE.Mesh(pinDotGeo, goldMat);
    pinDot.position.set(-0.28, 0.26, -0.28);
    pcbGroup.add(pinDot);

    // Gold corner mounting pads
    const padPositions = [
      [-1.0, 0.13, -0.7],
      [1.0, 0.13, -0.7],
      [-1.0, 0.13, 0.7],
      [1.0, 0.13, 0.7],
    ];
    padPositions.forEach(([px, py, pz]) => {
      const padGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.02, 16);
      const pad = new THREE.Mesh(padGeo, goldMat);
      pad.position.set(px, py, pz);
      pcbGroup.add(pad);
    });

    scene.add(pcbGroup);

    // 8. Fixed Coordinate Reference Axes (X=Red, Y=Green, Z=Blue)
    const origin = new THREE.Vector3(0, 0.25, 0);
    const axisLen = 1.3;
    const xAxis = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), origin, axisLen, 0xcc3333, 0.2, 0.1);
    const yAxis = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), origin, axisLen, 0x33aa33, 0.2, 0.1);
    const zAxis = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), origin, axisLen, 0x3366cc, 0.2, 0.1);
    scene.add(xAxis);
    scene.add(yAxis);
    scene.add(zAxis);

    // 9. Dynamic Acceleration Vector Arrow (Terracotta)
    const norm = Math.max(magnitude, 0.001);
    const dir = new THREE.Vector3(ax / norm, ay / norm, az / norm);
    const arrowLen = Math.min(Math.max(magnitude * 0.22, 0.4), 3.2);
    const arrow = new THREE.ArrowHelper(dir, origin, arrowLen, 0xc4603a, 0.35, 0.18);
    scene.add(arrow);
    arrowRef.current = arrow;

    // 10. Animation Loop
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 11. Resize Handler
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      camera.aspect = newWidth / h;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, h);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      controls.dispose();
      renderer.dispose();
      scene.clear();
    };
  }, [height]);

  // Update dynamic arrow when ax, ay, az change
  useEffect(() => {
    if (!arrowRef.current) return;
    const origin = new THREE.Vector3(0, 0.25, 0);
    const norm = Math.max(magnitude, 0.001);
    const dir = new THREE.Vector3(ax / norm, ay / norm, az / norm);
    const arrowLen = Math.min(Math.max(magnitude * 0.22, 0.4), 3.2);

    arrowRef.current.setDirection(dir);
    arrowRef.current.setLength(arrowLen, 0.35, 0.18);

    // State dependent color
    const colorHex =
      activityName === 'Walking' ? 0x5b8266 : activityName === 'Sitting' ? 0xc98a2c : 0xc4603a;
    arrowRef.current.setColor(colorHex);
  }, [ax, ay, az, magnitude, activityName]);

  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(4.5, 3.8, 5.0);
      controlsRef.current.target.set(0, 0.2, 0);
      controlsRef.current.update();
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: `${height}px`,
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
        border: '1px solid var(--border-sand)',
      }}
    >
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Floating Readout Overlay */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          backgroundColor: 'rgba(255, 252, 247, 0.88)',
          backdropFilter: 'blur(4px)',
          border: '1px solid var(--border-sand)',
          borderRadius: 'var(--radius-xs)',
          padding: '8px 12px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.78rem',
          color: 'var(--text-primary)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ fontWeight: 700, color: 'var(--accent-terracotta)', marginBottom: '3px' }}>
          3D Triaxial Vector Readout
        </div>
        <div>
          a<sub>x</sub>: <span style={{ color: '#cc3333' }}>{ax >= 0 ? `+${ax.toFixed(2)}` : ax.toFixed(2)}</span> m/s²
        </div>
        <div>
          a<sub>y</sub>: <span style={{ color: '#33aa33' }}>{ay >= 0 ? `+${ay.toFixed(2)}` : ay.toFixed(2)}</span> m/s²
        </div>
        <div>
          a<sub>z</sub>: <span style={{ color: '#3366cc' }}>{az >= 0 ? `+${az.toFixed(2)}` : az.toFixed(2)}</span> m/s²
        </div>
        <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid var(--border-light)', fontWeight: 600 }}>
          ||a||: {magnitude.toFixed(2)} m/s²
        </div>
      </div>

      {/* Control Buttons */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          right: '12px',
          display: 'flex',
          gap: '6px',
        }}
      >
        <Button
          variant="secondary"
          size="sm"
          onClick={handleResetCamera}
          title="Reset Camera Angle"
          style={{
            backgroundColor: 'rgba(255, 252, 247, 0.9)',
            boxShadow: 'var(--shadow-sm)',
            padding: '6px 10px',
          }}
        >
          <RotateCcw size={13} />
          <span style={{ fontSize: '0.75rem' }}>Reset View</span>
        </Button>
      </div>

      {/* Axis Color Key */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          display: 'flex',
          gap: '10px',
          fontSize: '0.72rem',
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-secondary)',
          backgroundColor: 'rgba(255, 252, 247, 0.85)',
          padding: '4px 8px',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-light)',
        }}
      >
        <span><strong style={{ color: '#cc3333' }}>●</strong> X-axis</span>
        <span><strong style={{ color: '#33aa33' }}>●</strong> Y-axis</span>
        <span><strong style={{ color: '#3366cc' }}>●</strong> Z-axis</span>
      </div>
    </div>
  );
};
