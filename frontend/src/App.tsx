import React, { useEffect, useState } from 'react';
import Login from './pages/Login';
import Register from './pages/Register';
import DocumentUpload from './pages/DocumentUpload';
import DocumentList from './pages/DocumentList';
import QueryInterface from './pages/QueryInterface';
import UserProfile from './pages/UserProfile';
import { getMe, removeToken } from './utils/api';

const App: React.FC = () => {
  const [user, setUser] = useState<any>(null);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [loading, setLoading] = useState(true);
  const [refreshDocs, setRefreshDocs] = useState(0);
  const [page, setPage] = useState<'dashboard' | 'profile'>('dashboard');

  useEffect(() => {
    getMe().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);

  const handleAuth = () => {
    setLoading(true);
    getMe().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  };

  const handleLogout = () => {
    removeToken();
    setUser(null);
  };

  if (loading) return <div className="flex items-center justify-center min-h-screen">Loading...</div>;

  if (!user) {
    return authMode === 'login' ? (
      <>
        <Login onLogin={handleAuth} />
        <div className="text-center mt-2">
          <button className="text-blue-600 underline" onClick={() => setAuthMode('register')}>No account? Register</button>
        </div>
      </>
    ) : (
      <>
        <Register onRegister={() => setAuthMode('login')} />
        <div className="text-center mt-2">
          <button className="text-blue-600 underline" onClick={() => setAuthMode('login')}>Already have an account? Login</button>
        </div>
      </>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex justify-between items-center p-4 bg-white shadow">
        <div className="font-bold">Notebook LLM</div>
        <div>
          <button
            className={`mr-4 ${page === 'dashboard' ? 'font-bold underline' : ''}`}
            onClick={() => setPage('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={`mr-4 ${page === 'profile' ? 'font-bold underline' : ''}`}
            onClick={() => setPage('profile')}
          >
            Profile
          </button>
          <button onClick={handleLogout} className="text-red-600">Logout</button>
        </div>
      </div>
      <div className="p-8">
        {page === 'dashboard' ? (
          <>
            <h1 className="text-2xl font-bold mb-4">Welcome, {user.email || 'User'}!</h1>
            <DocumentUpload onUpload={() => setRefreshDocs(r => r + 1)} />
            <DocumentList key={refreshDocs} />
            <div className="my-8 border-t" />
            <QueryInterface />
          </>
        ) : (
          <UserProfile />
        )}
      </div>
    </div>
  );
};

export default App;
