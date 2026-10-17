import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, roc_curve, precision_recall_curve, auc

class Evaluator:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_prob=None, num_classes=4):
        """Calculates Accuracy, Precision, Recall, F1 Score, and Specificity."""
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        # Calculate Specificity per class and average
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        specificities = []
        for i in range(num_classes):
            temp_cm = np.delete(np.delete(cm, i, axis=0), i, axis=1)
            tn = temp_cm.sum()
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        avg_spec = np.mean(specificities)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "specificity": float(avg_spec)
        }
        return metrics, cm

    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """Calculates MAE, RMSE, and R2 score."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        }

    @staticmethod
    def save_confusion_matrix(cm, classes, output_path="outputs/results/confusion_matrix.png"):
        """Plots and saves a high-quality 300 DPI confusion matrix."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="AgriClaim-XAI Confusion Matrix",
               ylabel='Ground Truth Label',
               xlabel='Predicted Label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_roc_pr_curves(y_true, y_prob_dict, output_dir="outputs/figures/", num_classes=4):
        """Generates publication-quality multiclass ROC and PR curves."""
        os.makedirs(output_dir, exist_ok=True)
        
        # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5), dpi=300)
        for name, y_prob in y_prob_dict.items():
            # Compute ROC for class 1 (or micro average)
            fpr, tpr, _ = roc_curve(y_true == 1, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)')
        ax_roc.legend(loc="lower right")
        fig_roc.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_roc)

        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5), dpi=300)
        for name, y_prob in y_prob_dict.items():
            precision, recall, _ = precision_recall_curve(y_true == 1, y_prob[:, 1])
            pr_auc = auc(recall, precision)
            ax_pr.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.2f})')
            
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall (PR) Curve')
        ax_pr.legend(loc="lower left")
        fig_pr.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_pr)
