import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UrbanSim WM - Smart City World Model",
  description: "Simulate and visualize urban system dynamics - energy, air quality, and mobility",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha512-p6EOvC4E7bY6dkprdFMn/hTyTX0bY4Z1cdqUPVHtV6nRvWmvMRGsiE9z1FMvx6bMpiKFFitvolGg5hPGfL5B7Q=="
          crossOrigin=""
        />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}

