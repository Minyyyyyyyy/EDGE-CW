// next.config.ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  images: {
    domains: ['yourazurestorage.blob.core.windows.net'], // Replace with your actual Azure domain
  },
};

export default nextConfig;
