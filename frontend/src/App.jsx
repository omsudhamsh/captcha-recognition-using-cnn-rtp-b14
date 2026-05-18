import { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
      setPrediction('');
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setPrediction('');
    setError('');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || 'Failed to predict captcha');
      }
    } catch (err) {
      setError('Error connecting to the server. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Captcha Recognition Project</h1>
          <a
            href="https://github.com/omsudhamsh"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
          >
            github.com/omsudhamsh
          </a>
        </div>
      </header>

      <main className="main-content">
        <div className="glass-card">
          <h2>Upload Captcha Image</h2>
          <p className="subtitle">Uses a Convolutional Neural Network to decode text from images.</p>

          <div className="upload-container">
            <input
              type="file"
              id="file-upload"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="file-upload" className="file-label">
              {selectedFile ? 'Change Image' : 'Choose an Image'}
            </label>
          </div>

          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Captcha Preview" className="preview-image" />
            </div>
          )}

          <button
            className={`predict-button ${loading ? 'loading' : ''} ${!selectedFile ? 'disabled' : ''}`}
            onClick={handlePredict}
            disabled={!selectedFile || loading}
          >
            {loading ? 'Processing...' : 'Recognize Captcha'}
          </button>

          {prediction && (
            <div className="result-container success">
              <h3>Recognition Result</h3>
              <div className="prediction-box">
                {prediction}
              </div>
            </div>
          )}

          {error && (
            <div className="result-container error">
              <p>{error}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
