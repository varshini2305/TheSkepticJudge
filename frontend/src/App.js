import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { HomeIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import AgentChat from './components/AgentChat';
import EvaluationDashboard from './components/EvaluationDashboard';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        {/* Navigation */}
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <h1 className="text-xl font-bold text-primary-600">The Skeptic Judge</h1>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link
                    to="/"
                    className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-900"
                  >
                    <HomeIcon className="h-5 w-5 mr-1" />
                    Chat
                  </Link>
                  <Link
                    to="/evaluation"
                    className="inline-flex items-center px-1 pt-1 text-sm font-medium text-gray-500 hover:text-gray-900"
                  >
                    <ChartBarIcon className="h-5 w-5 mr-1" />
                    Evaluation
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<AgentChat />} />
            <Route path="/evaluation" element={<EvaluationDashboard />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 