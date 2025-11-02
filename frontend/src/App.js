import './App.css';
import SymptomForm from './components/SymptomForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">🩺</div>
            <div className="header-title">
              <h1>Disease Prediction System</h1>
              <span className="header-subtitle">AI-Powered Health Assessment</span>
            </div>
          </div>
          <nav className="header-nav">
            <a href="#home" className="nav-link">Home</a>
            <a href="#predict" className="nav-link">Predict</a>
          </nav>
        </div>
      </header>

      <main>
        <section className="hero-section" id="home">
          <h2>Welcome to Health Assistant</h2>
          <p>
            Get instant disease predictions based on your symptoms using advanced 
            machine learning algorithms trained on comprehensive medical data.
          </p>
        </section>

        <section className="stats-section">
          <div className="stat-card">
            <div className="stat-number">41</div>
            <div className="stat-label">Diseases Detected</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">132</div>
            <div className="stat-label">Symptoms Analyzed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">97%+</div>
            <div className="stat-label">Accuracy Rate</div>
          </div>
        </section>

        <section className="features-section">
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>Accurate Predictions</h3>
            <p>
              Powered by XGBoost and ensemble machine learning models 
              trained on thousands of medical cases.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Instant Results</h3>
            <p>
              Get disease predictions in seconds. Simply select your 
              symptoms and receive immediate analysis.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3>Secure & Private</h3>
            <p>
              Your health data is processed securely and never stored 
              or shared with third parties.
            </p>
          </div>
        </section>

        <section id="predict">
          <SymptomForm />
        </section>
      </main>

      <footer>
        <p>© 2025 Disease Prediction System | Powered by Machine Learning</p>
        <p style={{ fontSize: '0.875rem', marginTop: '0.5rem', color: '#718096' }}>
          ⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.
        </p>
      </footer>
    </div>
  );
}

export default App;
