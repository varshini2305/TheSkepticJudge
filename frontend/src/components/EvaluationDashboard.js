import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const EVALUATOR_URL = process.env.REACT_APP_EVALUATOR_URL || 'http://localhost:8002';

const EvaluationDashboard = () => {
  const [selectedAgent, setSelectedAgent] = useState('learning_agent');
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchEvaluationResults = useCallback(async () => {
    try {
      const response = await axios.get(`${EVALUATOR_URL}/evaluation/summary/${selectedAgent}`);
      console.log('API Response:', response.data);
      if (response.data && response.data.summary) {
        setEvaluationResults(response.data.summary);
      }
    } catch (err) {
      console.error('Error fetching evaluation results:', err);
      console.error('Error details:', err.response?.data);
    }
  }, [selectedAgent]);

  useEffect(() => {
    fetchEvaluationResults();
  }, [fetchEvaluationResults]);

  const evaluateAgent = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('Running evaluation for agent:', selectedAgent);
      const response = await axios.post(`${EVALUATOR_URL}/evaluate`, {
        agent_name: selectedAgent
      });
      console.log('Evaluation response:', response.data);
      await fetchEvaluationResults();
    } catch (err) {
      console.error('Evaluation error:', err);
      console.error('Error details:', err.response?.data);
      setError(err.response?.data?.detail || 'An error occurred during evaluation');
    } finally {
      setLoading(false);
    }
  };

  const renderMetrics = () => {
    if (!evaluationResults) {
      return (
        <div className="text-center py-8 text-gray-500">
          Loading evaluation results...
        </div>
      );
    }

    const { average_scores, tool_usage_stats, latency_stats } = evaluationResults;

    return (
      <div className="space-y-6">
        {/* Average Scores */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Average Scores</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(average_scores || {}).map(([metric, score]) => (
              <div key={metric} className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500 capitalize">{metric.replace(/_/g, ' ')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {score.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Tool Usage */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Tool Usage</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(tool_usage_stats || {}).map(([tool, stats]) => (
              <div key={tool} className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500 capitalize">{tool.replace(/_/g, ' ')}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {stats.total_uses} uses
                </p>
                <p className="text-sm text-gray-500">
                  Success Rate: {((stats.successful_uses / stats.total_uses) * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Latency Statistics */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Latency Statistics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Average Latency</p>
              <p className="text-2xl font-semibold text-gray-900">
                {(latency_stats?.average || 0).toFixed(2)}ms
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Min Latency</p>
              <p className="text-2xl font-semibold text-gray-900">
                {(latency_stats?.min || 0).toFixed(2)}ms
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Max Latency</p>
              <p className="text-2xl font-semibold text-gray-900">
                {(latency_stats?.max || 0).toFixed(2)}ms
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Agent Evaluation Dashboard</h3>
          
          {/* Agent Selection and Evaluation Controls */}
          <div className="mt-4 flex items-center space-x-4">
            <select
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              className="block w-48 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
            >
              <option value="learning_agent">Learning Agent</option>
              <option value="restaurant_agent">Restaurant Agent</option>
            </select>
            
            <button
              onClick={evaluateAgent}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              {loading ? 'Evaluating...' : 'Run Evaluation'}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-400 p-4">
              <div className="flex">
                <div className="ml-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Metrics Display */}
          <div className="mt-6">
            {renderMetrics()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EvaluationDashboard; 