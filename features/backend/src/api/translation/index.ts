import express from 'express';
import { urduHandler } from './urdu';
import { validateHandler } from './validate';

const router = express.Router();

router.post('/urdu', urduHandler);
router.post('/validate', validateHandler);
router.get('/status', (req, res) => {
  res.json({
    status: 'available',
    supportedLanguages: ['en', 'ur'],
    serviceHealth: 'operational',
    lastUpdate: new Date().toISOString()
  });
});

export default router;