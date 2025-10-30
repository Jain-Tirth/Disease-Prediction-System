import './App.css';
import SymptomForm from './components/SymptomForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Disease Prediction</h1>
      </header>
      <main>
        <SymptomForm />
      </main>
    </div>
  );
}

export default App;
