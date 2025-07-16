import React, { useState } from 'react';
import { uploadDocument } from '../utils/api';

const DocumentUpload: React.FC<{ onUpload: () => void }> = ({ onUpload }) => {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError('');
    setSuccess(false);
    try {
      await uploadDocument(file);
      setSuccess(true);
      onUpload();
    } catch (err) {
      setError('Upload failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded shadow mb-6 max-w-md mx-auto">
      <h2 className="text-xl font-bold mb-4">Upload Document</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          onChange={e => setFile(e.target.files?.[0] || null)}
          className="mb-2 w-full"
          required
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          disabled={loading || !file}
        >
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </form>
      {error && <div className="text-red-500 mt-2">{error}</div>}
      {success && <div className="text-green-600 mt-2">Upload successful!</div>}
    </div>
  );
};

export default DocumentUpload; 