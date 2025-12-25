import React, { useEffect } from 'react';
import clsx from 'clsx';
import styles from './CyberpunkRobot.module.css';

const CyberpunkRobot = ({ size = 'medium', animation = true, color = 'primary' }) => {
  useEffect(() => {
    // Add any dynamic effects here if needed
    const robots = document.querySelectorAll('.cyberpunk-robot');
    robots.forEach(robot => {
      // Add hover effects
      robot.addEventListener('mouseenter', () => {
        robot.classList.add('cyberpunk-robot--hover');
      });

      robot.addEventListener('mouseleave', () => {
        robot.classList.remove('cyberpunk-robot--hover');
      });
    });

    return () => {
      robots.forEach(robot => {
        robot.removeEventListener('mouseenter', () => {});
        robot.removeEventListener('mouseleave', () => {});
      });
    };
  }, []);

  const getSizeClass = () => {
    switch(size) {
      case 'small':
        return styles.robotSmall;
      case 'large':
        return styles.robotLarge;
      default:
        return styles.robotMedium;
    }
  };

  const getColorClass = () => {
    switch(color) {
      case 'secondary':
        return styles.robotSecondary;
      case 'accent':
        return styles.robotAccent;
      default:
        return styles.robotPrimary;
    }
  };

  return (
    <div className={clsx(
      styles.robotContainer,
      getSizeClass(),
      getColorClass(),
      {[styles.robotAnimated]: animation}
    )}>
      <div className={styles.robot}>
        {/* Robot head */}
        <div className={clsx(styles.robotPart, styles.robotHead)}>
          {/* Robot eyes */}
          <div className={clsx(styles.robotPart, styles.robotEye, styles.robotEyeLeft)}></div>
          <div className={clsx(styles.robotPart, styles.robotEye, styles.robotEyeRight)}></div>
          {/* Robot antenna */}
          <div className={clsx(styles.robotPart, styles.robotAntenna)}></div>
        </div>

        {/* Robot body */}
        <div className={clsx(styles.robotPart, styles.robotBody)}>
          {/* Body details */}
          <div className={styles.robotBodyDetails}></div>
        </div>

        {/* Robot arms */}
        <div className={clsx(styles.robotPart, styles.robotArm, styles.robotArmLeft)}>
          <div className={styles.robotHand}></div>
        </div>
        <div className={clsx(styles.robotPart, styles.robotArm, styles.robotArmRight)}>
          <div className={styles.robotHand}></div>
        </div>

        {/* Robot legs */}
        <div className={clsx(styles.robotPart, styles.robotLeg, styles.robotLegLeft)}></div>
        <div className={clsx(styles.robotPart, styles.robotLeg, styles.robotLegRight)}></div>
      </div>

      {/* Robot glow effect */}
      <div className={styles.robotGlow}></div>

      {/* Robot scan lines */}
      <div className={styles.robotScanLines}></div>
    </div>
  );
};

export default CyberpunkRobot;