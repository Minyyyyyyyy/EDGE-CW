"use client";

import React, { useState, useEffect } from 'react';
import { AlertCircle, RefreshCw, CheckCircle, Filter, Camera, Info, MapPin } from 'lucide-react';

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [fallDetected, setFallDetected] = useState(false);
  const [timestamp, setTimestamp] = useState('');
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterLocation, setFilterLocation] = useState('all');
  const [filterTime, setFilterTime] = useState('today');
  const [showFilters, setShowFilters] = useState(false);
  const [currentTime, setCurrentTime] = useState('');

  // Set the current time once after component mounts to avoid hydration mismatch
  useEffect(() => {
    setCurrentTime(new Date().toLocaleString());
  }, []);

  // Fetch images from the backend API
  const fetchImages = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/screenshots');
      const data = await res.json();

      if (data.screenshots && data.screenshots.length > 0) {
        setImages(data.screenshots);
        // Set the first image as selected
        setSelectedImage(data.screenshots[0]);
        setFallDetected(data.screenshots[0].fallDetected);
        setTimestamp(data.screenshots[0].timestamp);
      } else {
        setImages([]);
        setSelectedImage(null);
      }
    } catch (error) {
      console.error('Error fetching screenshots:', error);
    }
    setLoading(false);
  };

  // Fetch images on initial load and set up periodic refresh
  useEffect(() => {
    fetchImages();
    const interval = setInterval(fetchImages, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchImages();
    setCurrentTime(new Date().toLocaleString());
  };

  const handleImageSelect = (image) => {
    setSelectedImage(image);
    setFallDetected(image.fallDetected);
    setTimestamp(image.timestamp);
  };

  const handleTestAlert = async () => {
    // Dummy base64 screenshot string for testing
    const dummyBase64Screenshot =
      "iVBORw0KGgoAAAANSUhEUgAAAAUA" +
      "AAAFCAYAAACNbyblAAAAHElEQVQI12P4" +
      "//8/w38GIAXDIBKE0DHxgljNBAAO" +
      "9TXL0Y4OHwAAAABJRU5ErkJggg==";

    const testAlertData = {
      screenshot: dummyBase64Screenshot,
      deviceId: 'test-device',
      timestamp: new Date().toISOString(),
    };

    try {
      const res = await fetch('/api/alerts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(testAlertData),
      });
      const data = await res.json();
      console.log('Test alert response:', data);
      fetchImages();
    } catch (error) {
      console.error('Error sending test alert:', error);
    }
  };

  // Apply filters to the images
  const filteredImages = images.filter(image => {
    let matchesStatus = filterStatus === 'all' ||
      (filterStatus === 'falls' && image.fallDetected) ||
      (filterStatus === 'normal' && !image.fallDetected);

    let matchesLocation = filterLocation === 'all' ||
      image.location.toLowerCase().includes(filterLocation.toLowerCase());

    // Simple time filtering logic (would need more sophisticated implementation in real app)
    let matchesTime = true; // Default to showing all for this example

    return matchesStatus && matchesLocation && matchesTime;
  });

  // Format timestamp for better readability
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b sticky top-0 z-10">
        <div className="container mx-auto py-5 px-6 max-w-7xl flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
            <Camera className="text-blue-600" />
            Fall Detection Dashboard
          </h1>
          <div className="flex gap-4">
            <button
              onClick={handleRefresh}
              className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-5 rounded-lg flex items-center gap-2 transition-colors"
            >
              <RefreshCw size={18} />
              <span>Refresh</span>
            </button>
            <button
              onClick={handleTestAlert}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-5 rounded-lg transition-colors"
            >
              Test Alert
            </button>
          </div>
        </div>
      </header>

      <div className="container mx-auto py-8 px-6 max-w-7xl">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main content - Selected image and gallery */}
          <div className="flex-grow order-2 lg:order-1">
            {/* Status indicator */}
            {selectedImage && (
              <div className={`mb-8 rounded-xl overflow-hidden shadow-lg transition-all duration-300 transform ${fallDetected ? 'scale-102 border-2 border-red-500' : ''}`}>
                <div className={`p-5 font-medium flex justify-between items-center ${fallDetected ? 'bg-red-600 text-white' : 'bg-white'}`}>
                  <div className="flex items-center gap-3">
                    {fallDetected ? (
                      <AlertCircle size={22} className="text-white" />
                    ) : (
                      <CheckCircle size={22} className="text-green-500" />
                    )}
                    <div>
                      <span className="font-bold text-lg">{fallDetected ? 'FALL DETECTED' : 'Normal Activity'}</span>
                      <span className="text-sm ml-3 opacity-80">
                        {formatTimestamp(selectedImage.timestamp)}
                      </span>
                    </div>
                  </div>
                  {fallDetected && (
                    <div className="animate-pulse">
                      <span className="bg-white text-red-600 text-sm py-1 px-4 rounded-full font-bold">
                        Alert
                      </span>
                    </div>
                  )}
                </div>
                <div className="bg-white">
                  <div className="relative">
                    <img
                      src={selectedImage.url}
                      alt={`Detection at ${selectedImage.timestamp}`}
                      className="w-full h-auto"
                    />
                    <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent text-white">
                      <div className="flex items-center gap-2">
                        <MapPin size={16} />
                        <span>{selectedImage.location}</span>
                      </div>
                    </div>
                    {fallDetected && (
                      <div className="absolute inset-0 border-4 border-red-500 pointer-events-none"></div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Image gallery with hover effects */}
            <div className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200">
              <div className="bg-gray-100 p-5 font-medium border-b flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <Camera size={20} className="text-gray-600" />
                  <span className="text-lg">Detection Gallery</span>
                  {filteredImages.length > 0 && (
                    <span className="bg-gray-200 text-gray-700 text-xs py-1 px-3 rounded-full ml-2">
                      {filteredImages.length} images
                    </span>
                  )}
                </div>
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="p-2 rounded-full hover:bg-gray-200 transition-colors"
                >
                  <Filter size={20} className="text-gray-600" />
                </button>
              </div>

              {/* Mobile filters (shown when toggle is active) */}
              {showFilters && (
                <div className="p-5 bg-gray-50 border-b lg:hidden">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
                      <select
                        className="w-full border rounded-lg p-3 text-sm"
                        value={filterStatus}
                        onChange={(e) => setFilterStatus(e.target.value)}
                      >
                        <option value="all">All Images</option>
                        <option value="falls">Falls Only</option>
                        <option value="normal">Normal Only</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Location</label>
                      <select
                        className="w-full border rounded-lg p-3 text-sm"
                        value={filterLocation}
                        onChange={(e) => setFilterLocation(e.target.value)}
                      >
                        <option value="all">All Locations</option>
                        <option value="area1">Area 1</option>
                        <option value="area2">Area 2</option>
                        <option value="area3">Area 3</option>
                        <option value="area4">Area 4</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              <div className="p-6 bg-white">
                {loading ? (
                  <div className="text-center py-16 text-gray-500">
                    <div className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                    <p>Loading detection images...</p>
                  </div>
                ) : filteredImages.length === 0 ? (
                  <div className="text-center py-16 text-gray-500">
                    <div className="w-20 h-20 rounded-full bg-gray-200 flex items-center justify-center mx-auto mb-4">
                      <Camera size={32} className="text-gray-400" />
                    </div>
                    <p className="text-lg">No detection images found</p>
                    <p className="text-sm mt-2">Try changing your filters or refreshing</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
                    {filteredImages.map((image) => (
                      <div
                        key={image.id}
                        className={`cursor-pointer rounded-lg overflow-hidden shadow-sm border-2 transition-all duration-200 hover:shadow-md transform hover:scale-105 ${
                          selectedImage && selectedImage.id === image.id
                            ? 'border-blue-500 ring-2 ring-blue-200'
                            : image.fallDetected
                              ? 'border-red-200 hover:border-red-400'
                              : 'border-transparent hover:border-blue-200'
                        }`}
                        onClick={() => handleImageSelect(image)}
                      >
                        <div className="relative">
                          <img
                            src={image.url}
                            alt={`Detection at ${image.timestamp}`}
                            className="w-full aspect-video object-cover"
                          />
                          {image.fallDetected && (
                            <div className="absolute top-2 right-2">
                              <span className="bg-red-600 text-white text-xs py-1 px-3 rounded-full font-medium flex items-center gap-1">
                                <AlertCircle size={12} />
                                Fall
                              </span>
                            </div>
                          )}
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 hover:opacity-100 transition-opacity flex items-end">
                            <div className="p-3 text-white text-xs w-full">
                              <div className="font-medium truncate">{formatTimestamp(image.timestamp)}</div>
                            </div>
                          </div>
                        </div>
                        <div className="p-3 bg-white">
                          <div className="flex items-center gap-2 text-xs text-gray-600">
                            <MapPin size={12} />
                            <span className="truncate">{image.location}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar - Status panel and filters */}
          <div className="lg:w-96 order-1 lg:order-2">
            {selectedImage && (
              <div className={`rounded-xl overflow-hidden shadow-lg mb-8 ${fallDetected ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
                <div className={`p-5 font-medium flex items-center gap-3 ${fallDetected ? 'text-red-800' : 'text-green-800'}`}>
                  {fallDetected ? <AlertCircle size={20} /> : <CheckCircle size={20} />}
                  <span className="text-lg">Alert Status</span>
                </div>
                <div className="p-5">
                  <div className="text-center">
                    {fallDetected ? (
                      <>
                        <div className="text-red-600 font-bold text-xl mb-4 flex items-center justify-center gap-3">
                          <AlertCircle size={24} />
                          FALL DETECTED
                        </div>
                        <div className="bg-red-100 rounded-lg p-4 mb-5">
                          <div className="text-sm text-red-800">
                            <div className="font-medium mb-2">Alert Details:</div>
                            <div className="flex justify-between text-sm">
                              <span>Timestamp:</span>
                              <span>{formatTimestamp(timestamp)}</span>
                            </div>
                            <div className="flex justify-between text-sm mt-2">
                              <span>Location:</span>
                              <span>{selectedImage.location}</span>
                            </div>
                          </div>
                        </div>
                        <button className="w-full bg-red-600 hover:bg-red-700 text-white py-3 px-4 rounded-lg font-medium transition-colors">
                          Acknowledge Alert
                        </button>
                      </>
                    ) : (
                      <>
                        <div className="text-green-600 font-bold text-xl mb-4 flex items-center justify-center gap-3">
                          <CheckCircle size={24} />
                          NO FALL DETECTED
                        </div>
                        <div className="bg-green-100 rounded-lg p-4 mb-3">
                          <div className="text-sm text-green-800">
                            <div className="font-medium mb-2">Status:</div>
                            <p className="text-sm">
                              Regular monitoring image captured during routine surveillance. No incidents detected.
                            </p>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Enhanced filters card */}
            <div className="bg-white rounded-xl overflow-hidden shadow-lg mb-8 border border-gray-200">
              <div className="bg-gray-100 p-5 font-medium border-b flex items-center gap-3">
                <Filter size={20} className="text-gray-600" />
                <span className="text-lg">Filter Detection Images</span>
              </div>
              <div className="p-5">
                <div className="space-y-5">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Status
                    </label>
                    <select
                      className="w-full border rounded-lg p-3 text-sm"
                      value={filterStatus}
                      onChange={(e) => setFilterStatus(e.target.value)}
                    >
                      <option value="all">All Images</option>
                      <option value="falls">Falls Only</option>
                      <option value="normal">Normal Only</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Location
                    </label>
                    <select
                      className="w-full border rounded-lg p-3 text-sm"
                      value={filterLocation}
                      onChange={(e) => setFilterLocation(e.target.value)}
                    >
                      <option value="all">All Locations</option>
                      <option value="area1">Area 1</option>
                      <option value="area2">Area 2</option>
                      <option value="area3">Area 3</option>
                      <option value="area4">Area 4</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Time Period
                    </label>
                    <select
                      className="w-full border rounded-lg p-3 text-sm"
                      value={filterTime}
                      onChange={(e) => setFilterTime(e.target.value)}
                    >
                      <option value="today">Today</option>
                      <option value="yesterday">Yesterday</option>
                      <option value="week">Past Week</option>
                      <option value="month">Past Month</option>
                    </select>
                  </div>

                  <button className="w-full bg-blue-500 hover:bg-blue-600 text-white py-3 px-4 rounded-lg font-medium transition-colors mt-2">
                    Apply Filters
                  </button>
                </div>
              </div>
            </div>

            {/* Enhanced system info card */}
            <div className="bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200">
              <div className="bg-gray-100 p-5 font-medium border-b flex items-center gap-3">
                <Info size={20} className="text-gray-600" />
                <span className="text-lg">System Information</span>
              </div>
              <div className="p-5">
                <ul className="space-y-4">
                  <li className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      System Status:
                    </span>
                    <span className="font-medium text-green-600 bg-green-50 px-3 py-1 rounded">Online</span>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-gray-600">Last update:</span>
                    <span className="font-medium text-sm">
                      {currentTime ? currentTime : 'Loading...'}
                    </span>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-gray-600">Total images:</span>
                    <span className="font-medium">{images.length}</span>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-gray-600">Falls detected:</span>
                    <div>
                      <span className="font-medium">{images.filter(img => img.fallDetected).length}</span>
                      <span className="text-xs text-gray-500 ml-2">
                        ({images.length > 0 ?
                          Math.round((images.filter(img => img.fallDetected).length / images.length) * 100) : 0}%)
                      </span>
                    </div>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-gray-600">Active cameras:</span>
                    <span className="font-medium">4</span>
                  </li>
                </ul>

                <div className="mt-6 pt-4 border-t border-gray-100">
                  <div className="text-sm text-gray-500 mb-2">System Health</div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div className="bg-green-500 h-3 rounded-full" style={{ width: '92%' }}></div>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>92% Operational</span>
                    <span>Updated: 2 min ago</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}