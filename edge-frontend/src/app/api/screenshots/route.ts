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
// import { NextResponse } from 'next/server';
// import path from 'path';
// import fs from 'fs';
//
// const IMAGE_FOLDER = path.join(process.cwd(), 'public', 'uploads');  // Store images in public/uploads
//
// export async function GET() {
//   try {
//     // Get all image files from the uploads folder
//     const files = fs.readdirSync(IMAGE_FOLDER);
//
//     let screenshots = [];
//
//     // Generate image URLs
//     files.forEach((file, index) => {
//       const filePath = path.join(IMAGE_FOLDER, file);
//       screenshots.push({
//         id: file,
//         url: `/uploads/${file}`,  // Make sure files are public
//         timestamp: new Date().toLocaleString(),
//         fallDetected: index % 2 === 0, // Just to simulate a fall detection
//         location: `Location ${Math.floor(index / 2) + 1}`,
//       });
//     });
//
//     return NextResponse.json({ screenshots });
//   } catch (error) {
//     console.error('Error reading local screenshots:', error);
//     return NextResponse.json({ error: 'Failed to fetch screenshots' }, { status: 500 });
//   }
// }
//

// src/app/api/screenshots/route.ts
import { NextResponse } from 'next/server';
import { S3Client, ListObjectsV2Command } from '@aws-sdk/client-s3';
console.log("AWS_REGION:", process.env.AWS_REGION);

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

const BUCKET_NAME = process.env.AWS_BUCKET_NAME;

export async function GET() {
  try {
    const params = { Bucket: BUCKET_NAME! };
    const command = new ListObjectsV2Command(params);
    const data = await s3Client.send(command);

    let screenshots = [];
    if (data.Contents) {
      screenshots = data.Contents.map((item) => {
        // Construct the public URL for the S3 object
        const url = `https://${BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${item.Key}`;
        return {
          id: item.Key,
          url,
          timestamp: item.LastModified ? new Date(item.LastModified).toLocaleString() : '',
          fallDetected: true,
          location: 'Camera 1',
        };
      });
    }

    return NextResponse.json({ screenshots });
  } catch (error) {
    console.error('Error listing screenshots from S3:', error);
    return NextResponse.json({ error: 'Failed to fetch screenshots' }, { status: 500 });
  }
}
