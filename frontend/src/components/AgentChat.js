import React, { useState, useEffect } from 'react';
import axios from 'axios';

const LEARNING_AGENT_URL = process.env.REACT_APP_LEARNING_AGENT_URL || 'http://localhost:8001';
const RESTAURANT_AGENT_URL = process.env.REACT_APP_RESTAURANT_AGENT_URL || 'http://localhost:8000';

const AgentChat = () => {
  const [selectedAgent, setSelectedAgent] = useState('learning_agent');
  const [query, setQuery] = useState('how does edge computing work?');
  const [userId, setUserId] = useState('aiden');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Update user ID and query when agent changes
  useEffect(() => {
    if (selectedAgent === 'learning_agent') {
      setUserId('aiden');
      setQuery('how does edge computing work?');
    } else {
      setUserId('alice_smith');
      setQuery('I want a vegan dinner tonight at 6pm');
    }
  }, [selectedAgent]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const endpoint = selectedAgent === 'learning_agent' 
        ? `${LEARNING_AGENT_URL}/learn`
        : `${RESTAURANT_AGENT_URL}/recommend`;

      const payload = {
        query,
        user_id: userId
      };

      const response = await axios.post(endpoint, payload);
      setResponse(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Chat with Agents</h3>
          
          {/* Agent Selection */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700">Select Agent</label>
            <select
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
            >
              <option value="learning_agent">Learning Agent</option>
              <option value="restaurant_agent">Restaurant Agent</option>
            </select>
          </div>

          {/* User ID Input */}
          <div className="mt-4">
            <label htmlFor="userId" className="block text-sm font-medium text-gray-700">
              User ID
            </label>
            <input
              type="text"
              id="userId"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
              placeholder="Enter your user ID"
            />
          </div>

          {/* Query Input */}
          <form onSubmit={handleSubmit} className="mt-4">
            <div>
              <label htmlFor="query" className="block text-sm font-medium text-gray-700">
                Your Query
              </label>
              <div className="mt-1">
                <textarea
                  id="query"
                  name="query"
                  rows={3}
                  className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your query here..."
                />
              </div>
            </div>
            <div className="mt-4">
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                {loading ? 'Processing...' : 'Send Query'}
              </button>
            </div>
          </form>

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

          {/* Response Display */}
          {response && (
            <div className="mt-6">
              <h4 className="text-lg font-medium text-gray-900">Response</h4>
              <div className="mt-2 bg-gray-50 rounded-lg p-4">
                {selectedAgent === 'restaurant_agent' ? (
                  <>
                    <div className="space-y-4">
                      {response.recommendations?.map((rec, index) => (
                        <div key={index} className="bg-white p-4 rounded-lg shadow">
                          <h5 className="font-medium text-lg">{rec.name}</h5>
                          <p className="text-gray-600">{rec.reason}</p>
                          <div className="mt-2 text-sm text-gray-500">
                            <p>Address: {rec.address}</p>
                            <p>Rating: {rec.rating}</p>
                            <p>Price Range: {rec.price_range}</p>
                            <p>Cuisine: {rec.cuisine}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4 text-gray-700">
                      <p>{response.summary}</p>
                    </div>
                  </>
                ) : (
                  <pre className="whitespace-pre-wrap text-sm text-gray-700">
                    {JSON.stringify(response, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentChat; 