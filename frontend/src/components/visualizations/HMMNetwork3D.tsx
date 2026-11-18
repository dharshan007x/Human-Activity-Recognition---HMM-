import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw } from 'lucide-react';
import { Button } from '../common/Button';

interface HMMNetwork3DProps {
  height?: number;
  transitionMatrix?: number[][];
  pi?: number[];
  stateNames?: string[];
}

export const HMMNetwork3D: React.FC<HMMNetwork3DProps> = ({
  height = 380,
  transitionMatrix = [
    [0.98, 0.01, 0.01],
    [0.005, 0.99, 0.005],
    [0.01, 0.005, 0.985],
  ],
  pi = [0.333, 0.333, 0.334],
  stateNames = ['Walking', 'Sitting', 'Running'],
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const [hoveredInfo, setHoveredInfo] = useState<string | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const h = height;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#F3EADB');

    const camera = new THREE.PerspectiveCamera(45, width / h, 0.1, 100);
    camera.position.set(0, 4.5, 6.5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 15;
    controls.minDistance = 3;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xfff8ee, 0.9);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 8, 5);
    scene.add(dirLight);

    // Ground circle
    const groundGeo = new THREE.CircleGeometry(4, 32);
    const groundMat = new THREE.MeshBasicMaterial({
      color: 0xeae0cf,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.2;
    scene.add(ground);

    // Positions of the 3 state nodes in a triangle
    const radius = 2.4;
    const angles = [
      Math.PI / 2, // Walking (Top)
      Math.PI / 2 + (2 * Math.PI) / 3, // Sitting (Bottom Left)
      Math.PI / 2 + (4 * Math.PI) / 3, // Running (Bottom Right)
    ];

    const nodePositions: THREE.Vector3[] = angles.map(
      (a) => new THREE.Vector3(radius * Math.cos(a), 0, radius * Math.sin(a))
    );

    const nodeColors = [0x5b8266, 0xc98a2c, 0xc4603a];

    // Create 3 State Spheres
    nodePositions.forEach((pos, idx) => {
      const priorWeight = pi[idx] || 0.333;
      const sphereRadius = 0.45 + priorWeight * 0.3;
      const sphereGeo = new THREE.SphereGeometry(sphereRadius, 32, 32);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: nodeColors[idx],
        roughness: 0.3,
        metalness: 0.2,
      });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.copy(pos);
      sphere.userData = { name: stateNames[idx], idx, type: 'node' };
      scene.add(sphere);

      // Self-loop torus (representing A_ii persistence)
      const selfProb = transitionMatrix[idx] ? transitionMatrix[idx][idx] : 0.98;
      const loopGeo = new THREE.TorusGeometry(0.35, 0.04 + selfProb * 0.03, 16, 32);
      const loopMat = new THREE.MeshStandardMaterial({
        color: nodeColors[idx],
        roughness: 0.4,
      });
      const loopMesh = new THREE.Mesh(loopGeo, loopMat);
      loopMesh.position.set(pos.x * 1.25, pos.y + 0.6, pos.z * 1.25);
      loopMesh.rotation.x = Math.PI / 4;
      scene.add(loopMesh);
    });

    // Create directed curved connections between distinct states (A_ij)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (i !== j) {
          const p1 = nodePositions[i];
          const p2 = nodePositions[j];
          const transProb = transitionMatrix[i] ? transitionMatrix[i][j] : 0.01;

          // Midpoint with curve arch
          const mid = new THREE.Vector3()
            .addVectors(p1, p2)
            .multiplyScalar(0.5)
            .add(new THREE.Vector3(0, 0.4 + (i + j) * 0.1, 0));

          const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
          const tubeGeo = new THREE.TubeGeometry(curve, 20, 0.03 + transProb * 0.15, 8, false);
          const tubeMat = new THREE.MeshStandardMaterial({
            color: 0x9c8c7e,
            roughness: 0.5,
            metalness: 0.1,
          });
          const tube = new THREE.Mesh(tubeGeo, tubeMat);
          tube.userData = {
            from: stateNames[i],
            to: stateNames[j],
            prob: transProb,
            type: 'edge',
          };
          scene.add(tube);
        }
      }
    }

    // Raycaster for hover inspection
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseMove = (event: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, false);

      if (intersects.length > 0) {
        const item = intersects[0].object.userData;
        if (item.type === 'node') {
          setHoveredInfo(`State: ${item.name} | Prior π = ${pi[item.idx].toFixed(4)}`);
        } else if (item.type === 'edge') {
          setHoveredInfo(`Transition A(${item.from} → ${item.to}) = ${item.prob.toFixed(4)}`);
        }
      } else {
        setHoveredInfo(null);
      }
    };
    container.addEventListener('mousemove', onMouseMove);

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
      container.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('resize', handleResize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      controls.dispose();
      renderer.dispose();
      scene.clear();
    };
  }, [height, transitionMatrix, pi, stateNames]);

  const handleResetCamera = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 4.5, 6.5);
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

      {/* Hover Information Banner */}
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
          fontSize: '0.82rem',
          color: 'var(--text-primary)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {hoveredInfo || 'Hover over states or transition arcs to inspect parameters'}
      </div>

      {/* State Node Key */}
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
          fontSize: '0.74rem',
        }}
      >
        <span style={{ color: '#5b8266', fontWeight: 600 }}>● S₁: Walking</span>
        <span style={{ color: '#c98a2c', fontWeight: 600 }}>● S₂: Sitting</span>
        <span style={{ color: '#c4603a', fontWeight: 600 }}>● S₃: Running</span>
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
