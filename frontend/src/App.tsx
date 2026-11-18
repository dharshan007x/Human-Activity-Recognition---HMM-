import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './components/layout/AppShell';

import { OverviewPage } from './pages/OverviewPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { PreprocessingPage } from './pages/PreprocessingPage';
import { SensorExplorerPage } from './pages/SensorExplorerPage';
import { HMMModelExplorerPage } from './pages/HMMModelExplorerPage';
import { ForwardPage } from './pages/ForwardPage';
import { BackwardPage } from './pages/BackwardPage';
import { PosteriorPage } from './pages/PosteriorPage';
import { RecognitionPage } from './pages/RecognitionPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { AboutPage } from './pages/AboutPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/dataset" element={<DatasetExplorerPage />} />
          <Route path="/preprocessing" element={<PreprocessingPage />} />
          <Route path="/sensor-explorer" element={<SensorExplorerPage />} />
          <Route path="/model-parameters" element={<HMMModelExplorerPage />} />
          <Route path="/forward" element={<ForwardPage />} />
          <Route path="/backward" element={<BackwardPage />} />
          <Route path="/posterior" element={<PosteriorPage />} />
          <Route path="/recognition" element={<RecognitionPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
