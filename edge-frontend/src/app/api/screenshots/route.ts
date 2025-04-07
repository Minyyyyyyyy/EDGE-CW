// // src/app/api/screenshots/route.ts
// import { NextResponse } from 'next/server';
// import { BlobServiceClient } from '@azure/storage-blob';
//
// const AZURE_STORAGE_CONNECTION_STRING = process.env.AZURE_STORAGE_CONNECTION_STRING;
// const CONTAINER_NAME = process.env.AZURE_CONTAINER_NAME || 'screenshots';
//
// export async function GET() {
//   try {
//     const blobServiceClient = BlobServiceClient.fromConnectionString(AZURE_STORAGE_CONNECTION_STRING!);
//     const containerClient = blobServiceClient.getContainerClient(CONTAINER_NAME);
//
//     let screenshots = [];
//     // Iterate over the blobs in the container
//     for await (const blob of containerClient.listBlobsFlat()) {
//       // Construct the public URL (this works if your container access is set to 'Blob')
//       const url = `${containerClient.url}/${blob.name}`;
//       // For demonstration, we'll include dummy metadata values.
//       screenshots.push({
//         id: blob.name,
//         url,
//         timestamp: new Date().toLocaleString(), // In a real system, you’d store metadata with each image.
//         fallDetected: true,  // Assume that every image uploaded represents a fall.
//         location: 'Unknown'
//       });
//     }
//
//     return NextResponse.json({ screenshots });
//   } catch (error) {
//     console.error('Error listing screenshots:', error);
//     return NextResponse.json({ error: 'Failed to fetch screenshots' }, { status: 500 });
//   }
// }

// src/app/api/screenshots/route.ts
import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';

const IMAGE_FOLDER = path.join(process.cwd(), 'public', 'uploads');  // Store images in public/uploads

export async function GET() {
  try {
    // Get all image files from the uploads folder
    const files = fs.readdirSync(IMAGE_FOLDER);

    let screenshots = [];

    // Generate image URLs
    files.forEach((file, index) => {
      const filePath = path.join(IMAGE_FOLDER, file);
      screenshots.push({
        id: file,
        url: `/uploads/${file}`,  // Make sure files are public
        timestamp: new Date().toLocaleString(),
        fallDetected: index % 2 === 0, // Just to simulate a fall detection
        location: `Location ${Math.floor(index / 2) + 1}`,
      });
    });

    return NextResponse.json({ screenshots });
  } catch (error) {
    console.error('Error reading local screenshots:', error);
    return NextResponse.json({ error: 'Failed to fetch screenshots' }, { status: 500 });
  }
}

