import React, { useState, useEffect } from 'react';
import { queryDocuments, getQueryHistory } from '../utils/api';

function renderResultContent(content: string) {
  // Image URL
  if (content.match(/\.(jpg|jpeg|png|gif)$/i)) {
    return <img src={content} alt="result" className="max-w-xs max-h-40 border" />;
  }
  // Code block (triple backticks or looks like code)
  if (/```[\s\S]*?```/.test(content)) {
    const code = content.replace(/```[a-zA-Z]*\n?|```/g, '');
    return <pre className="bg-gray-100 p-2 rounded overflow-x-auto text-sm"><code>{code}</code></pre>;
  }
  if (/^\s*([a-zA-Z0-9_]+\s*=|def |class |import |#include |function |public |private |var |let |const )/m.test(content)) {
    return <pre className="bg-gray-100 p-2 rounded overflow-x-auto text-sm"><code>{content}</code></pre>;
  }
  // Table (CSV/TSV)
  if (/^(?:[\w\s]+,)+[\w\s]+\n(?:[\w\s]+,)+[\w\s]+/m.test(content)) {
    const rows = content.trim().split(/\r?\n/);
    const cells = rows.map(row => row.split(','));
    return (
      <table className="border mt-2 mb-2">
        <tbody>
          {cells.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j} className="border px-2 py-1">{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    );
  }
  // HTML Table
  if (content.trim().startsWith('<table')) {
    return <div className="overflow-x-auto" dangerouslySetInnerHTML={{ __html: content }} />;
  }
  // Markdown (basic)
  if (content.includes('# ') || content.includes('**')) {
    // Simple markdown to HTML (bold, headers)
    let html = content
      .replace(/^# (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h4>$1</h4>')
      .replace(/\*\*(.*?)\*\*/gim, '<b>$1</b>')
      .replace(/\n/g, '<br/>');
    return <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />;
  }
  // Default: plain text (truncate if long)
  return <span>{content.length > 200 ? content.slice(0, 200) + '...' : content}</span>;
}

const QueryInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState('');

  const fetchHistory = () => {
    setHistoryLoading(true);
    setHistoryError('');
    getQueryHistory()
      .then(data => setHistory(data.history || []))
      .catch(() => setHistoryError('Failed to load query history.'))
      .finally(() => setHistoryLoading(false));
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await queryDocuments(query);
      setResult(res);
      fetchHistory();
    } catch (err) {
      setError('Query failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded shadow max-w-2xl mx-auto mt-8">
      <h2 className="text-xl font-bold mb-4">Ask a Question</h2>
      <form onSubmit={handleSubmit} className="mb-4">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          className="w-3/4 p-2 border rounded mr-2"
          placeholder="Enter your research question..."
          required
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? 'Querying...' : 'Ask'}
        </button>
      </form>
      {error && <div className="text-red-500 mb-2">{error}</div>}
      {result && (
        <div className="mt-4">
          <div className="mb-2"><span className="font-semibold">Answer:</span> {renderResultContent(result.answer)}</div>
          <div className="mb-2"><span className="font-semibold">Subqueries:</span> {Array.isArray(result.subqueries) ? result.subqueries.join(', ') : result.subqueries}</div>
          <div className="mb-2"><span className="font-semibold">Modality:</span> {result.modality}</div>
          <div className="mb-2"><span className="font-semibold">Results:</span>
            <ul className="list-disc ml-6">
              {result.results?.map((r: string, i: number) => (
                <li key={i} className="mb-1">
                  {renderResultContent(r)}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-2">Query History</h3>
        {historyLoading ? (
          <div>Loading history...</div>
        ) : historyError ? (
          <div className="text-red-500">{historyError}</div>
        ) : history.length === 0 ? (
          <div>No previous queries.</div>
        ) : (
          <ul className="list-disc ml-6">
            {history.map((item, idx) => (
              <li key={idx} className="mb-2">
                <div className="font-semibold">{item.query || JSON.stringify(item)}</div>
                {item.answer && <div className="text-gray-700">{renderResultContent(item.answer)}</div>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default QueryInterface; 