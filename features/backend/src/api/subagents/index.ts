import express from 'express';
import { registerHandler, listHandler, updateHandler, deleteHandler } from './management';

const router = express.Router();

router.post('/register', registerHandler);
router.get('/', listHandler);
router.put('/:id', updateHandler);
router.delete('/:id', deleteHandler);

export default router;