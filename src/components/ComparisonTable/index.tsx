import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type ComparisonItem = {
  traditional: string;
  physical: string;
};

const ComparisonData: ComparisonItem[] = [
  {
    traditional: "Screen-based interfaces",
    physical: "Embodied intelligence systems"
  },
  {
    traditional: "Digital-only constraints",
    physical: "Real-world physics understanding"
  },
  {
    traditional: "Simulated physics (optional)",
    physical: "Sensor fusion and perception"
  },
  {
    traditional: "No real-world interaction",
    physical: "Spatial reasoning required"
  },
  {
    traditional: "Focus on data processing",
    physical: "Human-robot interaction design"
  }
];

export default function ComparisonTable(): React.ReactElement {
  return (
    <section className={styles.comparisonSection}>
      <div className="container padding-vert--xl">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Traditional AI vs Physical AI</h2>
            <p className={styles.sectionSubtitle}>
              Understanding the fundamental differences between traditional AI development and Physical AI
            </p>

            <div className={styles.tableContainer}>
              <table className={clsx('comparison-table', styles.comparisonTable)}>
                <thead>
                  <tr>
                    <th className={clsx(styles.tableHeader, styles.traditionalHeader)}>
                      Traditional AI Development
                    </th>
                    <th className={clsx(styles.tableHeader, styles.physicalHeader)}>
                      Physical AI Development
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {ComparisonData.map((item, index) => (
                    <tr key={index} className={styles.tableRow}>
                      <td className={clsx(styles.tableCell, styles.traditionalCell)}>
                        {item.traditional}
                      </td>
                      <td className={clsx(styles.tableCell, styles.physicalCell)}>
                        {item.physical}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}