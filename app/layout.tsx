import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'STEMI Detector',
  description: 'Client-side STEMI screening using simple ML',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
