import express from 'express';
import { signupHandler } from './signup';
import { loginHandler } from './login';
import { meHandler } from './me';

const router = express.Router();

router.post('/signup', signupHandler);
router.post('/login', loginHandler);
router.get('/me', meHandler);

export default router;