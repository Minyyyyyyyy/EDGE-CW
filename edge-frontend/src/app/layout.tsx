// src/app/layout.tsx
import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Fall Detection Dashboard',
  description: 'Monitor fall detection alerts on your edge devices',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        {/* You can add meta tags or links here */}
      </head>
      <body>
        <header style={{ padding: '1rem', backgroundColor: '#f5f5f5' }}>
          <h1>Fall Detection Dashboard</h1>
        </header>
        <main style={{ padding: '2rem' }}>
          {children}
        </main>
        <footer style={{ padding: '1rem', backgroundColor: '#f5f5f5', textAlign: 'center' }}>
          © 2025 Your Project Name
        </footer>
      </body>
    </html>
  )
}
