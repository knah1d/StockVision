import React, { useState } from 'react';
import LLMService from '../services/llmService';
import ExplainModal from './ExplainModal';
import './ExplainButton.css';

const ExplainButton = ({ 
  chartData, 
  chartRef, 
  defaultQuestion = "Can you explain what this chart shows and what it means for beginners?",
  contextInfo = "",
  className = "",
  size = "medium" 
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [explanation, setExplanation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleExplain = async () => {
    setIsModalOpen(true);
    setLoading(true);
    setError(null);
    setExplanation('');

    try {
      // Create a comprehensive question based on context
      const fullQuestion = contextInfo 
        ? `${defaultQuestion}\n\nAdditional context: ${contextInfo}`
        : defaultQuestion;

      // Use the smart explanation method
      const result = await LLMService.explainChartData(
        chartData, 
        chartRef, 
        fullQuestion, 
        false // Start with Google AI, fallback to local if needed
      );
      
      setExplanation(result);
    } catch (err) {
      console.error('Explanation error:', err);
      setError('Sorry, I couldn\'t explain this chart right now. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setExplanation('');
    setError(null);
  };

  return (
    <>
      <button 
        onClick={handleExplain}
        className={`explain-button ${size} ${className}`}
        disabled={loading}
        title="Get AI explanation of this chart"
      >
        <span className="explain-icon">🤖</span>
        <span className="explain-text">Explain Chart</span>
      </button>

      <ExplainModal
        isOpen={isModalOpen}
        onClose={closeModal}
        explanation={explanation}
        loading={loading}
        error={error}
      />
    </>
  );
};

export default ExplainButton;
