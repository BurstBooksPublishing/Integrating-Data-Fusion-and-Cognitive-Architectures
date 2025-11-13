#include 
#include 

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("loaned_pub");
  // QoS: reliable, keep last with depth 4 to bound memory.
  rclcpp::QoS qos(4);
  qos.reliable();
  auto pub = node->create_publisher("camera/image", qos);

  rclcpp::WallRate loop_rate(30); // target publish rate
  while (rclcpp::ok()) {
    // borrow a loaned message to avoid allocation+copy
    auto loaned = pub->borrow_loaned_message();
    auto & msg = *loaned; // fill in-place
    msg.header.stamp = node->now();
    msg.height = 480; msg.width = 640;
    msg.encoding = "rgb8"; // small inline comment: set fields
    msg.data.resize(msg.height * msg.width * 3); // backstore uses shared memory in proper rmw
    // populate image bytes (omitted): hardware DMA can write directly into loaned buffer
    pub->publish(std::move(loaned)); // publish without an extra copy
    rclcpp::spin_some(node); // observe matched subscribers, QoS events
    loop_rate.sleep();
  }
  rclcpp::shutdown();
  return 0;
}