// src/app/api/screenshots/route.ts
import { NextResponse } from 'next/server';
import { S3Client, ListObjectsV2Command } from '@aws-sdk/client-s3';

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
      screenshots = data.Contents.filter((item) => item.Key.endsWith('.jpg'))  // Filter only .jpg files
        .map((item) => {
          // Construct the public URL for the S3 object
          const url = `https://${BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${item.Key}`;
          return {
            id: item.Key,
            url,
            timestamp: item.LastModified ? new Date(item.LastModified).toLocaleString() : '',
            fallDetected: true,  // Assuming fall detected for each image (adjust as per your logic)
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
