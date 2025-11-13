#include 
#include 
#include 
#include 
#include 
#include 

struct Frame { // small payload for demo
  uint64_t seq;
  uint64_t ts_ns;
  char data[256];
};

class SpscRing {
  std::vector pool_;
  const size_t size_;
  std::atomic head_{0}; // next write index (producer)
  std::atomic tail_{0}; // next read index (consumer)
public:
  explicit SpscRing(size_t n): pool_(n), size_(n) {}
  // Producer: attempt to claim a slot. Returns pointer or nullptr if full.
  Frame* try_produce() {
    size_t h = head_.load(std::memory_order_relaxed);
    size_t t = tail_.load(std::memory_order_acquire);
    if ((h + 1) % size_ == t) return nullptr; // full
    return &pool_[h];
  }
  // Producer: publish after DMA fills frame.
  void publish() {
    size_t h = head_.load(std::memory_order_relaxed);
    head_.store((h + 1) % size_, std::memory_order_release);
  }
  // Consumer: attempt to take a frame. Returns pointer or nullptr if empty.
  Frame* try_consume() {
    size_t t = tail_.load(std::memory_order_relaxed);
    size_t h = head_.load(std::memory_order_acquire);
    if (t == h) return nullptr; // empty
    return &pool_[t];
  }
  // Consumer: release frame after processing.
  void release() {
    size_t t = tail_.load(std::memory_order_relaxed);
    tail_.store((t + 1) % size_, std::memory_order_release);
  }
};

int main() {
  const size_t N = 1024; // ring size (power-of-two assists masking)
  SpscRing ring(N);
  const int total = 100000;
  std::atomic done{false};

  // Consumer thread
  std::thread consumer([&](){
    int received = 0;
    while (received < total) {
      Frame* f = ring.try_consume();
      if (!f) { std::this_thread::yield(); continue; }
      // process frame (cheap here)
      uint64_t latency_ns = std::chrono::duration_cast(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() - f->ts_ns;
      if (received % 20000 == 0) std::cout << "consumed seq=" << f->seq
                                           << " latency_ns=" << latency_ns << "\n";
      ring.release();
      ++received;
    }
    done = true;
  });

  // Producer thread (simulates DMA writing directly into frame)
  std::thread producer([&](){
    for (int i = 0; i < total; ++i) {
      while (true) {
        Frame* f = ring.try_produce();
        if (!f) { std::this_thread::yield(); continue; }
        // DMA would write directly into f; simulate by filling it here.
        f->seq = i;
        f->ts_ns = std::chrono::duration_cast(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::memset(f->data, 0xAB, sizeof(f->data)); // payload
        ring.publish();
        break;
      }
    }
  });

  producer.join();
  consumer.join();
  std::cout << "done\n";
  return 0;
}