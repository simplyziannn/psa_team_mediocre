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

  console.log(`[PROXY] Spawning python process for prompt: "${text.substring(0, 50)}..."`);
  const pyPath = path.resolve(__dirname, '../code/main.py');
  // Use -u to force unbuffered stdout/stderr so we capture output immediately
  const py = spawn('python', ['-u', pyPath], { stdio: ['pipe', 'pipe', 'pipe'] });

  let stdout = '';
  let stderr = '';
  py.stdout.on('data', (d) => { 
    console.log('[PROXY] Python stdout:', d.toString());
    stdout += d.toString(); 
  });
  py.stderr.on('data', (d) => { 
    console.error('[PROXY] Python stderr:', d.toString());
    stderr += d.toString(); 
  });

  py.on('close', (code) => {
    console.log(`[PROXY] Python process exited with code ${code}`);
    if (code !== 0) {
      return res.status(500).json({ ok: false, error: `Python script exited with code ${code}.`, stderr: stderr || 'No stderr output.' });
    }
    try {
      const parsed = JSON.parse(stdout);
      return res.json({ ok: true, result: parsed });
    } catch (e) {
      return res.status(500).json({ ok: false, error: 'Failed to parse JSON from Python script.', raw: stdout, stderr });
    }
  });

  // Send the text on stdin and close
  py.stdin.write(text);
  py.stdin.end();
});

app.listen(PORT, () => {
  console.log(`Frontend proxy server listening on http://localhost:${PORT}`);
});

