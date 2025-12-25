import express from 'express';
import { auth } from '../lib/auth';
import { authRouter } from './auth';
import { personalizationRouter } from './personalization';
import { translationRouter } from './translation';
import { subagentsRouter } from './subagents';

const router = express.Router();

// Mount Better Auth routes
router.use(auth.apiRoute);

// Mount custom API routes
router.use('/auth', authRouter);
router.use('/personalization', personalizationRouter);
router.use('/translation', translationRouter);
router.use('/subagents', subagentsRouter);

export default router;