import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw } from 'lucide-react';
import { Button } from '../common/Button';

interface PosteriorLandscape3DProps {
  height?: number;
  gamma: number[][]; // [T, 3]
  currentStep?: number;
  stateNames?: string[];
}

export const PosteriorLandscape3D: React.FC<PosteriorLandscape3DProps> = ({
  height = 380,
  gamma,
  currentStep = 0,
  stateNames = ['Walking', 'Sitting', 'Running'],
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const sliceMarkerRef = useRef<THREE.Mesh | null>(null);

  const T = gamma && gamma.length > 0 ? gamma.length : 50;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const h = height;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#F3EADB');

    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.set(5.5, 4.5, 6.0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 18;
    controls.minDistance = 2.5;
    controls.target.set(0, 0.5, 0);
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xfff8ee, 0.85);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(5, 8, 4);
    scene.add(dirLight);

    // Build the 3D probability mesh: PlaneGeometry with segments
    const meshWidth = 6.0;
    const meshDepth = 2.5;
    const geo = new THREE.PlaneGeometry(meshWidth, meshDepth, Math.max(T - 1, 2), 2);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    const colors: number[] = [];

    const stateColors = [
      new THREE.Color('#5B8266'), // Walking
      new THREE.Color('#C98A2C'), // Sitting
      new THREE.Color('#C4603A'), // Running
    ];

    for (let i = 0; i < pos.count; i++) {
      // Determine time index t and state index s from vertex index
      const tIdx = Math.min(Math.floor((i % (T || 1))), (gamma?.length || 1) - 1);
      const sIdx = Math.min(Math.floor(i / (T || 1)), 2);

      const prob = gamma && gamma[tIdx] ? gamma[tIdx][sIdx] : 0.33;
      pos.setY(i, prob * 1.5); // Probability height scale

      const baseColor = stateColors[sIdx];
      colors.push(baseColor.r, baseColor.g, baseColor.b);
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.35,
      metalness: 0.1,
      wireframe: false,
      side: THREE.DoubleSide,
    });
    const surfaceMesh = new THREE.Mesh(geo, mat);
    scene.add(surfaceMesh);

    // Wireframe overlay for editorial topographic contour look
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x4a3a2d,
      wireframe: true,
      transparent: true,
      opacity: 0.25,
    });
    const wireMesh = new THREE.Mesh(geo, wireMat);
    scene.add(wireMesh);

    // Baseline plane
    const baseGeo = new THREE.PlaneGeometry(meshWidth + 0.5, meshDepth + 0.5);
    baseGeo.rotateX(-Math.PI / 2);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0xe2d3bc, side: THREE.DoubleSide });
    const baseMesh = new THREE.Mesh(baseGeo, baseMat);
    baseMesh.position.y = -0.01;
    scene.add(baseMesh);

    // Slice Marker Needle for currentStep
    const sliceGeo = new THREE.BoxGeometry(0.08, 1.8, meshDepth + 0.2);
    const sliceMat = new THREE.MeshBasicMaterial({
      color: 0x2e2118,
      transparent: true,
      opacity: 0.65,
    });
    const sliceMarker = new THREE.Mesh(sliceGeo, sliceMat);
    sliceMarker.position.y = 0.9;
    scene.add(sliceMarker);
    sliceMarkerRef.current = sliceMarker;

    // Animation Loop
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
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
  }, [height, gamma, T]);

  // Update position of time-step slice needle
  useEffect(() => {
    if (!sliceMarkerRef.current) return;
    const meshWidth = 6.0;
    const tFrac = T > 1 ? currentStep / (T - 1) : 0;
    const xPos = -meshWidth / 2 + tFrac * meshWidth;
    sliceMarkerRef.current.position.x = xPos;
  }, [currentStep, T]);

  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(5.5, 4.5, 6.0);
      controlsRef.current.target.set(0, 0.5, 0);
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

      {/* Overlay Status */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          backgroundColor: 'rgba(255, 252, 247, 0.92)',
          padding: '6px 14px',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-sand)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          color: 'var(--text-primary)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <span>3D Posterior Landscape &gamma;<sub>t</sub>(i)</span>
        <span style={{ marginLeft: '10px', color: 'var(--accent-terracotta)' }}>
          Active Slice: t = {currentStep} / {T - 1}
        </span>
      </div>

      {/* Axis Guide */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          display: 'flex',
          gap: '12px',
          backgroundColor: 'rgba(255, 252, 247, 0.88)',
          padding: '4px 10px',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-light)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.72rem',
        }}
      >
        <span><strong>X:</strong> Time (0 &rarr; {T - 1})</span>
        <span><strong>Y:</strong> Posterior Mass (0 &rarr; 1.0)</span>
        <span><strong>Z:</strong> States ({stateNames.join(', ')})</span>
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
