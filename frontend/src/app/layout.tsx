import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Urban Object Detection - YOLOv8 Live",
  description:
    "Real-time urban object detection using YOLOv8 with ONNX Runtime Web. Detect cars, people, bicycles, traffic lights, and 37 urban object classes directly in your browser.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
