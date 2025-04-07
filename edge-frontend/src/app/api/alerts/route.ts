// // src/app/api/alerts/route.ts
// import { NextResponse } from 'next/server';
// import { BlobServiceClient } from '@azure/storage-blob';
//
// const AZURE_STORAGE_CONNECTION_STRING = process.env.AZURE_STORAGE_CONNECTION_STRING;
// const CONTAINER_NAME = process.env.AZURE_CONTAINER_NAME || 'screenshots';
//
// export async function POST(request: Request) {
//   try {
//     const { screenshot, deviceId, timestamp } = await request.json();
//
//     // Convert the base64 screenshot string to a Buffer
//     const imageBuffer = Buffer.from(screenshot, 'base64');
//     const filename = `fall_${deviceId}_${Date.now()}.jpg`;
//
//     // Create a BlobServiceClient to connect to your Storage Account
//     const blobServiceClient = BlobServiceClient.fromConnectionString(AZURE_STORAGE_CONNECTION_STRING!);
//     const containerClient = blobServiceClient.getContainerClient(CONTAINER_NAME);
//     const blockBlobClient = containerClient.getBlockBlobClient(filename);
//
//     // Upload the image buffer to Azure Blob Storage
//     await blockBlobClient.uploadData(imageBuffer, {
//       blobHTTPHeaders: { blobContentType: 'image/jpeg' },
//     });
//
//     console.log(`Screenshot uploaded as ${filename}`);
//     return NextResponse.json({ message: 'Alert received and screenshot uploaded successfully' });
//   } catch (error) {
//     console.error('Error uploading screenshot:', error);
//     return NextResponse.json({ error: 'Failed to process alert' }, { status: 500 });
//   }
// }

// src/app/api/alerts/route.ts
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const UPLOAD_FOLDER = path.join(process.cwd(), 'public', 'uploads');  // Image upload path

export async function POST(request: Request) {
  try {
    const { screenshot, deviceId, timestamp } = await request.json();
    // Convert the base64 string to a Buffer
    const imageBuffer = Buffer.from(screenshot, 'base64');
    const filename = `fall_${deviceId}_${Date.now()}.jpg`;

    // Ensure the upload folder exists
    if (!fs.existsSync(UPLOAD_FOLDER)) {
      fs.mkdirSync(UPLOAD_FOLDER);
    }

    // Write the image buffer to the local file system
    fs.writeFileSync(path.join(UPLOAD_FOLDER, filename), imageBuffer);

    console.log(`Screenshot uploaded locally as ${filename}`);

    return NextResponse.json({ message: 'Alert received and screenshot uploaded successfully' });
  } catch (error) {
    console.error('Error uploading screenshot locally:', error);
    return NextResponse.json({ error: 'Failed to process alert' }, { status: 500 });
  }
}
