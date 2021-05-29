const toggleBtn = document.querySelector('.navbar__toogleBtn');
const menu = document.querySelector('.navbar__menu');
const icons = document.querySelector('.navbar__icons');

toggleBtn.addEventListener('click', () => {
    menu.classList.toggle('acitve');
    icons.classList.toggle('acitve');
});

//Fecth the items from the JSON file
function loadproduct() {
    return fetch('data/data.json')  //fetch로 지정된 경로에서 데이터 받아오고
        .then(response => response.json())  //성공적으로 받으면 json으로 변환
        .then(json => json.items);  //json 안에 있는 items를 return
}
function createHTMLString(item) {
    return '<li> </li>'
        ;
}
//main
loadproduct()
    .then(item => {
        displayItems(items);//html에 item 보여주기
        setEventListeners(items)
    })
    .catch()


