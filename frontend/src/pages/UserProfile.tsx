import React, { useEffect, useState } from 'react';
import { getMe } from '../utils/api';

const UserProfile: React.FC = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    getMe()
      .then(setUser)
      .catch(() => setError('Failed to load user info.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading user info...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="bg-white p-6 rounded shadow max-w-md mx-auto mt-8">
      <h2 className="text-xl font-bold mb-4">User Profile</h2>
      <div className="mb-2"><span className="font-semibold">Email:</span> {user.email}</div>
      <div className="mb-2"><span className="font-semibold">ID:</span> {user.id}</div>
      {/* Add more fields as needed */}
    </div>
  );
};

export default UserProfile; 