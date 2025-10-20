const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(bodyParser.json({ limit: '2mb' }));

app.post('/api/process', (req, res) => {
  const text = (req.body && req.body.text) ? String(req.body.text) : '';
  if (!text) return res.status(400).json({ ok: false, error: 'text is required' });

  // Spawn the Python orchestrator. Run from frontend folder; main.py is in ../code/main.py
  const pyPath = path.resolve(__dirname, '../code/main.py');
  const py = spawn('python', [pyPath], { stdio: ['pipe', 'pipe', 'pipe'] });

  let stdout = '';
  let stderr = '';
  py.stdout.on('data', (d) => { stdout += d.toString(); });
  py.stderr.on('data', (d) => { stderr += d.toString(); });

  py.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ ok: false, error: `python exit ${code}`, stderr });
    }
    try {
      const parsed = JSON.parse(stdout);
      return res.json({ ok: true, result: parsed });
    } catch (e) {
      return res.status(500).json({ ok: false, error: 'invalid json from python', raw: stdout, stderr });
    }
  });

  // Send the text on stdin and close
  py.stdin.write(text);
  py.stdin.end();
});

app.listen(PORT, () => {
  console.log(`Frontend proxy server listening on http://localhost:${PORT}`);
});
