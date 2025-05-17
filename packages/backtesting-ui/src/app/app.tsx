// Uncomment this line to use CSS modules
// import styles from './app.module.scss';
import { Link, Route, Routes } from 'react-router-dom';
import SimpleChart from '../components/SimpleChart';

export function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-md p-4">
        <h1 className="text-2xl font-bold text-gray-800">Backtesting UI</h1>
      </header>

      <main className="container mx-auto p-4">
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Sample Visualization</h2>
          <SimpleChart title="Monthly Performance" />
        </div>

        {/* START: routes */}
        <div className="mt-8 p-4 bg-white rounded-lg shadow">
          <div role="navigation" className="mb-4">
            <ul className="flex space-x-4">
              <li>
                <Link to="/" className="text-blue-500 hover:text-blue-700">
                  Home
                </Link>
              </li>
              <li>
                <Link
                  to="/page-2"
                  className="text-blue-500 hover:text-blue-700"
                >
                  Page 2
                </Link>
              </li>
            </ul>
          </div>
          <Routes>
            <Route
              path="/"
              element={
                <div>
                  This is the generated root route.{' '}
                  <Link
                    to="/page-2"
                    className="text-blue-500 hover:text-blue-700"
                  >
                    Click here for page 2.
                  </Link>
                </div>
              }
            />
            <Route
              path="/page-2"
              element={
                <div>
                  <Link to="/" className="text-blue-500 hover:text-blue-700">
                    Click here to go back to root page.
                  </Link>
                </div>
              }
            />
          </Routes>
        </div>
        {/* END: routes */}
      </main>
    </div>
  );
}

export default App;
