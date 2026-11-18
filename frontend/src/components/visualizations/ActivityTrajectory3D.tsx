import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw } from 'lucide-react';
import { Button } from '../common/Button';

interface ActivityTrajectory3DProps {
  height?: number;
  initialMode?: 'all' | 'walking' | 'sitting' | 'running';
}

export const ActivityTrajectory3D: React.FC<ActivityTrajectory3DProps> = ({
  height = 380,
  initialMode = 'all',
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const [selectedActivity, setSelectedActivity] = useState<'all' | 'walking' | 'sitting' | 'running'>(
    initialMode
  );

  // Group references to toggle visibility
  const walkingGroupRef = useRef<THREE.Group | null>(null);
  const sittingGroupRef = useRef<THREE.Group | null>(null);
  const runningGroupRef = useRef<THREE.Group | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const h = height;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#F3EADB');

    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.set(6, 5, 8);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 20;
    controls.minDistance = 3;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xfff8ee, 0.85);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight.position.set(6, 10, 6);
    scene.add(dirLight);

    // Coordinate grid & axes
    const grid = new THREE.GridHelper(10, 20, 0xcbb99f, 0xe2d3bc);
    grid.position.y = -2;
    scene.add(grid);

    // 1. Synthetic / Empirical Walking Trajectory: Rhythmic loop
    const walkingGroup = new THREE.Group();
    const walkingPoints: THREE.Vector3[] = [];
    const N_walk = 200;
    for (let i = 0; i < N_walk; i++) {
      const theta = (i / N_walk) * Math.PI * 4;
      const x = 0.9 * Math.sin(theta) + 0.15 * Math.sin(2 * theta);
      const y = 1.6 * Math.cos(theta) + 0.3 * Math.cos(3 * theta);
      const z = 0.7 * Math.sin(2 * theta);
      walkingPoints.push(new THREE.Vector3(x, y, z));
    }
    const walkingCurve = new THREE.CatmullRomCurve3(walkingPoints, true);
    const walkingGeo = new THREE.TubeGeometry(walkingCurve, 120, 0.06, 8, true);
    const walkingMat = new THREE.MeshStandardMaterial({
      color: 0x5b8266, // Sage green
      roughness: 0.3,
      metalness: 0.1,
    });
    const walkingMesh = new THREE.Mesh(walkingGeo, walkingMat);
    walkingGroup.add(walkingMesh);

    // Add particle spheres along walking path
    for (let i = 0; i < walkingPoints.length; i += 8) {
      const dotGeo = new THREE.SphereGeometry(0.08, 8, 8);
      const dotMat = new THREE.MeshBasicMaterial({ color: 0x476b51 });
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(walkingPoints[i]);
      walkingGroup.add(dot);
    }
    scene.add(walkingGroup);
    walkingGroupRef.current = walkingGroup;

    // 2. Sitting Trajectory: Quiescent point cluster near (0, 0, 9.8) - normalized to (0, -0.5, 0)
    const sittingGroup = new THREE.Group();
    const sittingGeo = new THREE.SphereGeometry(0.35, 24, 24);
    const sittingMat = new THREE.MeshStandardMaterial({
      color: 0xc98a2c, // Ochre
      roughness: 0.4,
      metalness: 0.2,
    });
    const sittingCenter = new THREE.Mesh(sittingGeo, sittingMat);
    sittingCenter.position.set(-1.8, -0.6, -1.2);
    sittingGroup.add(sittingCenter);

    // Subtle micro-movement noise points around sitting center
    const sittingDotsGeo = new THREE.BufferGeometry();
    const sittingPos: number[] = [];
    for (let i = 0; i < 80; i++) {
      sittingPos.push(
        -1.8 + (Math.random() - 0.5) * 0.4,
        -0.6 + (Math.random() - 0.5) * 0.4,
        -1.2 + (Math.random() - 0.5) * 0.4
      );
    }
    sittingDotsGeo.setAttribute('position', new THREE.Float32BufferAttribute(sittingPos, 3));
    const sittingDotsMat = new THREE.PointsMaterial({ color: 0x9e6c21, size: 0.08 });
    const sittingPointsCloud = new THREE.Points(sittingDotsGeo, sittingDotsMat);
    sittingGroup.add(sittingPointsCloud);
    scene.add(sittingGroup);
    sittingGroupRef.current = sittingGroup;

    // 3. Running Trajectory: High energy, wide orbit loop (~2.8 Hz)
    const runningGroup = new THREE.Group();
    const runningPoints: THREE.Vector3[] = [];
    const N_run = 200;
    for (let i = 0; i < N_run; i++) {
      const theta = (i / N_run) * Math.PI * 4;
      const x = 2.4 * Math.sin(theta) + 0.4 * Math.sin(3 * theta);
      const y = 3.2 * Math.cos(theta) + 0.7 * Math.sin(2 * theta);
      const z = 1.9 * Math.cos(2 * theta);
      runningPoints.push(new THREE.Vector3(x, y, z));
    }
    const runningCurve = new THREE.CatmullRomCurve3(runningPoints, true);
    const runningGeo = new THREE.TubeGeometry(runningCurve, 140, 0.09, 8, true);
    const runningMat = new THREE.MeshStandardMaterial({
      color: 0xc4603a, // Terracotta
      roughness: 0.3,
      metalness: 0.2,
    });
    const runningMesh = new THREE.Mesh(runningGeo, runningMat);
    runningGroup.add(runningMesh);

    for (let i = 0; i < runningPoints.length; i += 6) {
      const dotGeo = new THREE.SphereGeometry(0.1, 8, 8);
      const dotMat = new THREE.MeshBasicMaterial({ color: 0x993d1f });
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.copy(runningPoints[i]);
      runningGroup.add(dot);
    }
    scene.add(runningGroup);
    runningGroupRef.current = runningGroup;

    // Animation Loop
    let clock = 0;
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      clock += 0.005;
      walkingGroup.rotation.y = clock * 0.4;
      runningGroup.rotation.y = clock * 0.4;
      sittingGroup.rotation.y = clock * 0.4;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      camera.aspect = newWidth / h;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      controls.dispose();
      renderer.dispose();
      scene.clear();
    };
  }, [height]);

  // Update visibility according to selected tab
  useEffect(() => {
    if (walkingGroupRef.current) {
      walkingGroupRef.current.visible = selectedActivity === 'all' || selectedActivity === 'walking';
    }
    if (sittingGroupRef.current) {
      sittingGroupRef.current.visible = selectedActivity === 'all' || selectedActivity === 'sitting';
    }
    if (runningGroupRef.current) {
      runningGroupRef.current.visible = selectedActivity === 'all' || selectedActivity === 'running';
    }
  }, [selectedActivity]);

  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(6, 5, 8);
      controlsRef.current.target.set(0, 0, 0);
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

      {/* Top Selector Bar */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          display: 'flex',
          gap: '4px',
          backgroundColor: 'rgba(255, 252, 247, 0.92)',
          padding: '4px',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-sand)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {(['all', 'walking', 'sitting', 'running'] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setSelectedActivity(mode)}
            style={{
              padding: '4px 10px',
              fontSize: '0.78rem',
              fontWeight: 600,
              border: 'none',
              borderRadius: 'var(--radius-xs)',
              cursor: 'pointer',
              backgroundColor:
                selectedActivity === mode ? 'var(--accent-terracotta)' : 'transparent',
              color: selectedActivity === mode ? '#FFFFFF' : 'var(--text-secondary)',
              textTransform: 'capitalize',
              transition: 'all 0.15s ease',
            }}
          >
            {mode === 'all' ? 'All Trajectories' : mode}
          </button>
        ))}
      </div>

      {/* Trajectory Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          backgroundColor: 'rgba(255, 252, 247, 0.88)',
          padding: '6px 12px',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-light)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.74rem',
          display: 'flex',
          gap: '12px',
        }}
      >
        <span style={{ color: '#5b8266', fontWeight: 600 }}>● Walking (Periodic ~1.8 Hz)</span>
        <span style={{ color: '#c98a2c', fontWeight: 600 }}>● Sitting (Static 1g Cluster)</span>
        <span style={{ color: '#c4603a', fontWeight: 600 }}>● Running (High Energy ~2.8 Hz)</span>
      </div>

      {/* Reset button */}
      <div style={{ position: 'absolute', bottom: '12px', right: '12px' }}>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleResetCamera}
          style={{
            backgroundColor: 'rgba(255, 252, 247, 0.9)',
            boxShadow: 'var(--shadow-sm)',
            padding: '6px 10px',
          }}
        >
          <RotateCcw size={13} />
          <span style={{ fontSize: '0.75rem' }}>Reset</span>
        </Button>
      </div>
    </div>
  );
};
