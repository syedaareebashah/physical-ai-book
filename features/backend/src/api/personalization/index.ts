import express from 'express';
import { processHandler } from './process';
import { preferencesHandler } from './preferences';

const router = express.Router();

router.post('/process', processHandler);
router.get('/preferences', preferencesHandler);
router.put('/preferences', preferencesHandler);

export default router;