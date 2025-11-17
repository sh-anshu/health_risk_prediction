const CACHE_NAME = 'dp-cache-v1';
const URLsToCache = ['/', '/static/css/main.css', '/static/js/main.js'];

self.addEventListener('install', (evt) => {
  evt.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(URLsToCache)));
  self.skipWaiting();
});

self.addEventListener('activate', (evt) => { evt.waitUntil(self.clients.claim()); });

self.addEventListener('fetch', (evt) => {
  evt.respondWith(caches.match(evt.request).then(r => r || fetch(evt.request)));
});
