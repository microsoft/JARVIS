<template>
  <div class="com__box">
    <div class="loading">
      <div></div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
    </div>
  </div>
</template>

<style scoped>
.loading,
.loading > div {
  position: relative;
  box-sizing: border-box;
}

.loading {
  display: block;
  font-size: 0;
  color: #000;
}

.loading.la-dark {
  color: #333;
}

.loading > div {
  display: inline-block;
  float: none;
  background-color: currentColor;
  border: 0 solid currentColor;
}

.loading {
  width: 17px;
  height: 17px;
}

.loading > div {
  width: 3px;
  height: 3px;
  margin: 1px;
  border-radius: 100%;
  animation-name: ball-grid-beat;
  animation-iteration-count: infinite;
}

.loading > div:nth-child(1) {
  animation-duration: 0.65s;
  animation-delay: 0.03s;
}

.loading > div:nth-child(2) {
  animation-duration: 1.02s;
  animation-delay: 0.09s;
}

.loading > div:nth-child(3) {
  animation-duration: 1.06s;
  animation-delay: -0.69s;
}

.loading > div:nth-child(4) {
  animation-duration: 1.5s;
  animation-delay: -0.41s;
}

.loading > div:nth-child(5) {
  animation-duration: 1.6s;
  animation-delay: 0.04s;
}

.loading > div:nth-child(6) {
  animation-duration: 0.84s;
  animation-delay: 0.07s;
}

.loading > div:nth-child(7) {
  animation-duration: 0.68s;
  animation-delay: -0.66s;
}

.loading > div:nth-child(8) {
  animation-duration: 0.93s;
  animation-delay: -0.76s;
}

.loading > div:nth-child(9) {
  animation-duration: 1.24s;
  animation-delay: -0.76s;
}

@keyframes ball-grid-beat {
  0% {
    opacity: 1;
  }

  50% {
    opacity: 0.35;
  }

  100% {
    opacity: 1;
  }
}
</style>
