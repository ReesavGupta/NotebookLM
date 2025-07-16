import React, { useEffect, useState } from 'react';
import { listDocuments, downloadDocument, deleteDocument } from '../utils/api';

const DocumentList: React.FC<{ refreshKey?: number }> = ({ refreshKey }) => {
  const [docs, setDocs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const fetchDocs = () => {
    setLoading(true);
    listDocuments()
      .then(setDocs)
      .catch(() => setError('Failed to load documents.'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchDocs();
    // eslint-disable-next-line
  }, [refreshKey]);

  const handleDownload = async (id: number) => {
    try {
      await downloadDocument(id);
    } catch (err) {
      alert('Download failed.');
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    setDeletingId(id);
    try {
      await deleteDocument(id);
      fetchDocs();
    } catch (err) {
      alert('Delete failed.');
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) return <div>Loading documents...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="bg-white p-6 rounded shadow max-w-2xl mx-auto">
      <h2 className="text-xl font-bold mb-4">Uploaded Documents</h2>
      {docs.length === 0 ? (
        <div>No documents uploaded yet.</div>
      ) : (
        <table className="w-full border">
          <thead>
            <tr>
              <th className="border px-2 py-1">ID</th>
              <th className="border px-2 py-1">Filename</th>
              <th className="border px-2 py-1">Uploaded By</th>
              <th className="border px-2 py-1">Created At</th>
              <th className="border px-2 py-1">Actions</th>
            </tr>
          </thead>
          <tbody>
            {docs.map(doc => (
              <tr key={doc.id}>
                <td className="border px-2 py-1">{doc.id}</td>
                <td className="border px-2 py-1">{doc.filename}</td>
                <td className="border px-2 py-1">{doc.uploaded_by || '-'}</td>
                <td className="border px-2 py-1">{doc.created_at || '-'}</td>
                <td className="border px-2 py-1">
                  <button
                    className="bg-green-600 text-white px-2 py-1 rounded mr-2 hover:bg-green-700"
                    onClick={() => handleDownload(doc.id)}
                  >
                    Download
                  </button>
                  <button
                    className="bg-red-600 text-white px-2 py-1 rounded hover:bg-red-700"
                    onClick={() => handleDelete(doc.id)}
                    disabled={deletingId === doc.id}
                  >
                    {deletingId === doc.id ? 'Deleting...' : 'Delete'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default DocumentList; 