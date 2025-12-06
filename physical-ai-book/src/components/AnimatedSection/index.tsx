import React, { useRef, useEffect, useState } from 'react';
import styles from './AnimatedSection.module.css';

const AnimatedSection = ({ children }) => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(sectionRef.current);
        }
      },
      {
        threshold: 0.1,
      }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => {
      if (sectionRef.current) {
        observer.unobserve(section.current);
      }
    };
  }, []);

  return (
    <section ref={sectionRef} className={`${styles.section} ${isVisible ? styles.visible : ''}`}>
      {children}
    </section>
  );
};

export default AnimatedSection;
