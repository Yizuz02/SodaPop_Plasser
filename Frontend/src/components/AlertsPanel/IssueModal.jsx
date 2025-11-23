import styles from "./IssueModal.module.css";

export default function IssueModal({ issue, onClose, onAccept }) {
  if (!issue) return null;

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h2>Issue #{issue.id}</h2>

        <div className={styles.detailsBox}>
          <p><strong>Lift left (mm):</strong> {issue.lift_left_mm}</p>
          <p><strong>Lift right (mm):</strong> {issue.lift_right_mm}</p>
          <p><strong>Adjustment left (mm):</strong> {issue.adjustment_left_mm}</p>
          <p><strong>Adjustment right (mm):</strong> {issue.adjustment_right_mm}</p>
        </div>

        <div className={styles.buttons}>
          <button className={styles.accept} onClick={onAccept}>Accept</button>
          <button className={styles.reject} onClick={onClose}>Reject</button>
        </div>
      </div>
    </div>
  );
}