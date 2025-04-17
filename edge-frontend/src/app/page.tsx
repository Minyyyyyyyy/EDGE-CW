"use client";

import React, { useState, useEffect } from 'react';
import { RefreshCw, AlertTriangle, CheckCircle, Clock, Map, Camera, Info, Filter } from 'lucide-react';

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [fallDetected, setFallDetected] = useState(false);
  const [timestamp, setTimestamp] = useState('');
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>("");

  // Fetch images from the backend API
  const fetchImages = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/screenshots');
      const data = await res.json();
      if (data.screenshots && data.screenshots.length > 0) {
        // Sort images by timestamp to show latest first
        const sortedImages = data.screenshots.sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        setImages(sortedImages);
        setSelectedImage(sortedImages[0]);
        setFallDetected(sortedImages[0].fallDetected);
        setTimestamp(sortedImages[0].timestamp);
      } else {
        setImages([]);
        setSelectedImage(null);
      }
    } catch (error) {
      console.error('Error fetching screenshots:', error);
    }
    setLoading(false);
  };

  // On initial mount, fetch images and set last update (client-only)
  useEffect(() => {
    fetchImages();
    setLastUpdate(new Date().toLocaleString());
    const interval = setInterval(() => {
      fetchImages();
      setLastUpdate(new Date().toLocaleString());
    }, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchImages();
    setLastUpdate(new Date().toLocaleString());
  };

  const handleImageSelect = (image: any) => {
    setSelectedImage(image);
    setFallDetected(image.fallDetected);
    setTimestamp(image.timestamp);
  };

  // Count falls detected
  const fallCount = images.filter((img: any) => img.fallDetected).length;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 flex flex-col">
      {/* Dark Header */}
      <header className="bg-black border-b border-gray-800 shadow-md">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center">
            <Camera className="mr-2 text-blue-400" size={28} />
            Fall Detection Dashboard
          </h1>
          <div className="flex items-center">
            <span className="hidden md:inline text-sm text-gray-400 mr-3">Last updated: {lastUpdate}</span>
            <button
              onClick={handleRefresh}
              className="flex items-center bg-gray-800 hover:bg-gray-700 transition-colors text-white text-sm py-2 px-4 rounded-md font-medium shadow-sm"
            >
              <RefreshCw size={16} className="mr-1" /> Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-4 md:p-6 flex-grow">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main content area - selected image */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-700">
              <div className={`p-4 font-medium flex justify-between items-center border-b ${
                selectedImage?.fallDetected
                  ? 'bg-gray-900 border-red-900'
                  : 'bg-gray-900 border-gray-700'
              }`}>
                <div className="flex items-center">
                  <Camera size={18} className="mr-2 text-gray-300" />
                  <span className="text-gray-200 font-semibold">Detection Details</span>
                </div>
                {selectedImage?.fallDetected && (
                  <div className="flex items-center bg-red-900 bg-opacity-50 text-red-300 py-1 px-3 rounded-full text-sm font-medium">
                    <AlertTriangle size={16} className="mr-1" /> Fall Detected
                  </div>
                )}
              </div>

              {selectedImage ? (
                <div className="p-0">
                  <div className="relative">
                    <img
                      src={selectedImage.url}
                      alt={`Detection at ${selectedImage.timestamp}`}
                      className="w-full h-auto"
                    />
                    {selectedImage.fallDetected && (
                      <div className="absolute top-4 right-4">
                        <span className="bg-red-600 text-white py-1 px-3 rounded-md font-medium shadow-lg animate-pulse">
                          FALL DETECTED
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="p-4 bg-gray-800">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="flex items-center">
                        <Clock size={18} className="mr-2 text-gray-400" />
                        <div>
                          <div className="text-sm text-gray-400">Timestamp</div>
                          <div className="font-medium text-gray-200">{selectedImage.timestamp}</div>
                        </div>
                      </div>
                      <div className="flex items-center">
                        <Map size={18} className="mr-2 text-gray-400" />
                        <div>
                          <div className="text-sm text-gray-400">Location</div>
                          <div className="font-medium text-gray-200">{selectedImage.location}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-gray-400">
                  No image selected
                </div>
              )}
            </div>

            {/* Image Gallery with hover effect */}
            <div className="mt-6 bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-700">
              <div className="p-4 font-medium border-b border-gray-700 bg-gray-900 flex justify-between items-center">
                <span className="text-gray-200 font-semibold">Recent Detections</span>
                {fallCount > 0 && (
                  <span className="bg-red-900 bg-opacity-50 text-red-300 py-1 px-3 rounded-full text-sm">
                    {fallCount} {fallCount === 1 ? 'Fall' : 'Falls'} Detected
                  </span>
                )}
              </div>

              <div className="p-4">
                {loading ? (
                  <div className="flex justify-center items-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                  </div>
                ) : images.length === 0 ? (
                  <div className="text-center py-12 text-gray-400">
                    <Camera size={48} className="mx-auto mb-4 text-gray-500" />
                    <p>No detection images available</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                    {images.map((image: any) => (
                      <div
                        key={image.id}
                        onClick={() => handleImageSelect(image)}
                        className={`cursor-pointer group rounded-lg overflow-hidden border transition-all hover:shadow-lg ${
                          selectedImage && selectedImage.id === image.id
                            ? 'ring-2 ring-blue-500 shadow-lg border-blue-500'
                            : 'border-gray-700 hover:border-gray-600'
                        }`}
                      >
                        <div className="relative">
                          <img
                            src={image.url}
                            alt={`Detection at ${image.timestamp}`}
                            className="w-full aspect-video object-cover transition-transform transform group-hover:scale-105"
                          />
                          {image.fallDetected && (
                            <div className="absolute top-2 right-2">
                              <span className="bg-red-600 text-white text-xs py-1 px-2 rounded-full shadow-sm">
                                Fall
                              </span>
                            </div>
                          )}
                          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all"></div>
                        </div>
                        <div className="p-2 bg-gray-900">
                          <div className="text-xs font-medium text-gray-300 truncate">{image.location}</div>
                          <div className="text-xs text-gray-400 truncate">{image.timestamp}</div>
                          {/* Show the label below the image */}
                          <div className="text-xs text-gray-500">{image.label}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            {/* Status card */}
            <div className={`bg-gray-800 rounded-xl shadow-lg overflow-hidden border ${
              fallDetected ? 'border-red-800' : 'border-green-800'
            }`}>
              <div className={`p-4 font-medium flex items-center ${
                fallDetected
                  ? 'bg-gray-900 text-red-400 border-b border-red-900'
                  : 'bg-gray-900 text-green-400 border-b border-green-900'
              }`}>
                {fallDetected ? (
                  <AlertTriangle size={18} className="mr-2" />
                ) : (
                  <CheckCircle size={18} className="mr-2" />
                )}
                <span className="font-semibold">Current Status</span>
              </div>

              <div className="p-6">
                <div className="flex flex-col items-center text-center">
                  {fallDetected ? (
                    <>
                      <div className="rounded-full bg-red-900 bg-opacity-30 p-4 mb-4">
                        <AlertTriangle size={32} className="text-red-500" />
                      </div>
                      <h3 className="text-red-500 font-bold text-xl mb-2">FALL DETECTED</h3>
                      <p className="text-gray-400 mb-4">Immediate attention required</p>
                      <div className="text-sm text-gray-500">
                        <p className="flex items-center justify-center">
                          <Clock size={14} className="mr-1" />
                          {timestamp}
                        </p>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="rounded-full bg-green-900 bg-opacity-30 p-4 mb-4">
                        <CheckCircle size={32} className="text-green-500" />
                      </div>
                      <h3 className="text-green-500 font-bold text-xl mb-2">ALL CLEAR</h3>
                      <p className="text-gray-400 mb-4">No falls detected</p>
                      <div className="text-sm text-gray-500">
                        <p className="flex items-center justify-center">
                          <Clock size={14} className="mr-1" />
                          Normal operation
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* System info card */}
            <div className="mt-6 bg-gray-800 rounded-xl shadow-lg overflow-hidden border border-gray-700">
              <div className="p-4 font-medium border-b border-gray-700 bg-gray-900">
                <span className="text-gray-200 font-semibold">System Information</span>
              </div>

              <div className="p-4">
                <ul className="space-y-4">
                  <li className="flex justify-between items-center px-2 py-2 hover:bg-gray-700 rounded transition-colors">
                    <div className="flex items-center text-gray-300">
                      <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                      Status
                    </div>
                    <span className="font-medium text-green-400">Online</span>
                  </li>
                  <li className="flex justify-between items-center px-2 py-2 hover:bg-gray-700 rounded transition-colors">
                    <div className="flex items-center text-gray-300">
                      <Clock size={16} className="mr-2 text-gray-400" />
                      Last Update
                    </div>
                    <span className="font-medium text-gray-200">{lastUpdate || 'Loading...'}</span>
                  </li>
                  <li className="flex justify-between items-center px-2 py-2 hover:bg-gray-700 rounded transition-colors">
                    <div className="flex items-center text-gray-300">
                      <Camera size={16} className="mr-2 text-gray-400" />
                      Total Images
                    </div>
                    <span className="font-medium text-gray-200">{images.length}</span>
                  </li>
                  <li className="flex justify-between items-center px-2 py-2 hover:bg-gray-700 rounded transition-colors">
                    <div className="flex items-center text-gray-300">
                      <AlertTriangle size={16} className="mr-2 text-gray-400" />
                      Falls Detected
                    </div>
                    <span className={`font-medium ${fallCount > 0 ? 'text-red-400' : 'text-gray-200'}`}>
                      {fallCount}
                    </span>
                  </li>
                </ul>

                <div className="mt-6">
                  <button
                    onClick={handleRefresh}
                    className="w-full flex items-center justify-center bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors py-2 px-4 rounded-md font-medium"
                  >
                    <RefreshCw size={16} className="mr-2" />
                    Refresh Dashboard
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Dark Footer */}
      <footer className="bg-black text-gray-400 border-t border-gray-800 mt-8">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <Camera size={18} className="mr-2 text-gray-500" />
              <span className="text-sm">Fall Detection System</span>
            </div>
            <div className="flex items-center text-sm">
              <div className="flex items-center mr-4">
                <Info size={14} className="mr-1 text-gray-500" />
                <span>Version 2.0.4</span>
              </div>
              <div className="text-gray-600">© {new Date().getFullYear()} Safety Monitoring</div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
