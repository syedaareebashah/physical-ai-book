import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type ModuleCardProps = {
  id: string;
  title: string;
  description: string;
  duration: string;
  level: string;
  completed?: boolean;
  onStart?: () => void;
  onReview?: () => void;
};

export default function ModuleCard({
  id,
  title,
  description,
  duration,
  level,
  completed = false,
  onStart,
  onReview
}: ModuleCardProps): JSX.Element {
  const handleClick = () => {
    if (completed && onReview) {
      onReview();
    } else if (onStart) {
      onStart();
    }
  };

  return (
    <div className={clsx('card', styles.moduleCard, completed ? styles.completed : '')}>
      <div className="card__header">
        <h3>{title}</h3>
        <span className={clsx('badge badge--secondary', styles.levelBadge, `badge--${level.toLowerCase()}`)}>
          {level}
        </span>
      </div>
      <div className="card__body">
        <p>{description}</p>
      </div>
      <div className="card__footer">
        <div className={styles.moduleFooter}>
          <span className="badge badge--info">{duration}</span>
          <button
            className={`button button--${completed ? 'secondary' : 'primary'} button--sm`}
            onClick={handleClick}
          >
            {completed ? 'Review Module' : 'Start Learning'}
          </button>
        </div>
      </div>
    </div>
  );
}