import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RV Single-Beat Analysis",
  description: "Right-ventricular single-beat pressure-volume analysis",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
