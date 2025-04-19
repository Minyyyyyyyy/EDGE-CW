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

    let audioFiles = [];
    if (data.Contents) {
      audioFiles = data.Contents.filter((item) => item.Key!.endsWith('.mp3')).map((item) => {
        // Construct the public URL for the S3 object
        const url = `https://${BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${item.Key}`;
        return {
          id: item.Key,
          url,
          timestamp: item.LastModified ? new Date(item.LastModified).toLocaleString() : '',
          location: 'Unknown', // You can modify this based on your application logic
        };
      });
    }

    return NextResponse.json({ audios: audioFiles });
  } catch (error) {
    console.error('Error listing audio files from S3:', error);
    return NextResponse.json({ error: 'Failed to fetch audio files' }, { status: 500 });
  }
}
