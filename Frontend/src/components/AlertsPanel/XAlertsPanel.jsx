import { useState } from "react";
import styles from "./AlertsPanel.module.css";
import IssueModal from "./IssueModal";

export default function AlertsPanel() {
  const [selectedIssue, setSelectedIssue] = useState(null);

  const issues = [
    {
      id: 1203,
      severity: "High",
      description: "Vertical lift required: 12 mm",
      location: "19.4326, -99.1332",
      date: "14:23",
      lift_left_mm: 5.2,
      lift_right_mm: 5.2,
      adjustment_left_mm: 3.1,
      adjustment_right_mm: 2.7,
    },
    {
      id: 1188,
      severity: "Medium",
      description: "Alignment deviation detected",
      location: "19.4371, -99.1290",
      date: "13:10",
      lift_left_mm: 2.1,
      lift_right_mm: 1.9,
      adjustment_left_mm: 1.0,
      adjustment_right_mm: 1.3,
    }
  ];

  const handleAccept = () => {
    alert(
      `✔ Machine automatically assigned to issue #${selectedIssue.id}`
    );
    setSelectedIssue(null);
  };

  return (
    <div className={styles.panel}>
      <h3 className={styles.title}>Active Alerts</h3>

      {issues.map((issue) => (
        <div key={issue.id} className={styles.alert}>
          <h4>Issue #{issue.id}</h4>
          <p><strong>Severity:</strong> {issue.severity}</p>
          <p><strong>Description:</strong> {issue.description}</p>
          <p><strong>Reported:</strong> {issue.date}</p>

          <button
            className={styles.btn}
            onClick={() => setSelectedIssue(issue)}
          >
            View Details
          </button>
        </div>
      ))}

      <IssueModal
        issue={selectedIssue}
        onClose={() => setSelectedIssue(null)}
        onAccept={handleAccept}
      />
    </div>
  );
}