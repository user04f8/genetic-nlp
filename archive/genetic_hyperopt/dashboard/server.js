import express from 'express';
import { readdir, readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Adjust the path to go up one level to repo-root and into /status
const STATUS_DIR = path.join(__dirname, '../status');

const app = express();

app.get('/status', (req, res) => {
  readdir(STATUS_DIR, (err, files) => {
    if (err) {
      res.status(500).send('Error reading status directory');
      return;
    }
    const statuses = [];
    files.forEach((file) => {
      if (file.endsWith('_status.json')) {
        const filePath = path.join(STATUS_DIR, file);
        const data = readFileSync(filePath, 'utf8');
        statuses.push(JSON.parse(data));
      }
    });
    res.json(statuses);
  });
});

app.listen(3001, () => {
  console.log('Server running on port 3001');
});
