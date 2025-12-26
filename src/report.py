# src/report.py

from fpdf import FPDF
import os

class DRReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Diabetic Retinopathy Classifier Report", ln=True, align="C")
        self.ln(10)

    def add_metrics(self, metrics):
        self.set_font("Arial", "", 12)
        self.cell(0, 10, "Evaluation Metrics:", ln=True)
        for key, value in metrics.items():
            self.cell(0, 10, f"{key.replace('_', ' ').title()}: {value:.4f}", ln=True)
        self.ln(10)

    def add_images(self, image_dir, title, prefix, max_images=5):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

        images = [f for f in os.listdir(image_dir) if f.startswith(prefix) and f.endswith(".png")]
        for img in images[:max_images]:
            path = os.path.join(image_dir, img)
            self.image(path, w=80)
            self.ln(5)

def generate_pdf_report(metrics, image_dir="visuals", output_path="reports/dr_report.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = DRReport()
    pdf.add_page()
    pdf.add_metrics(metrics)
    pdf.add_images(image_dir, "Grad-CAM Overlays", "gradcam_")
    pdf.add_images(image_dir, "SHAP Explanations", "shap_")
    pdf.output(output_path)
    print(f"✅ PDF report saved to {output_path}")
