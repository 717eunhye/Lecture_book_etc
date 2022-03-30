
***
**SSH로 Github 연동하기**
***
ssh로 github을 연동하지 않고 사용할 경우 매번 아이디와 패스워드를 입력하여 로그인해야하는 번거로움이 발생한다.   

또 여러 git 계정을 사용할 경우 ssh를 연동하지 않으면 정말 귀찮은 짓을 해야한다.. 

다음은 내가 몇 번의 실패를 겪고 정리한 내용으로 순서대로 따라하면 별 무리없이 잘 실행될 것이다.

1. Local repository 폴더를 생성한 후 git bash CLI를 열고, SSH key 생성
```
ssh-keygen -t rsa -b 4096 -C "github email 주소 입력"
```
- 알아보기 쉽게 **'My_home'** 명으로 파일명을 만들었지만 아래 이미지가 나올때까지 엔터를 쳐서 넘어가도 된다. 따로 지정을 하지 않으면 기본적으로 id_rsa, id_rsa.pub라는 이름으로 2개의 파일이 생성된다.
- .pub는 공개키로 github에 등록할 것이다.

<p align="center">  
<img src="https://user-images.githubusercontent.com/85601490/122715403-c87bf080-d2a3-11eb-8087-b5a4a2ea9f45.png" width="400"/>  
</p> 

 2. Remote repository에서 settings - SSH and GPG keys - New SSH key를 클릭하여 title은 개인이 알아보기 편하게 입력하고, key 부분에  pub 파일 내용을 복사하여 붙여넣는다.
```
cat "test_ntels.pub"
```
<p align="center">  
<img src="https://user-images.githubusercontent.com/85601490/122715969-a2a31b80-d2a4-11eb-877c-3a1ad355c205.png" width="400"/>  
</p> 

 3.  저장 후 ssh 실행 여부를 확인하고 테스트를 해보자.  git bash CLI창에서 아래와 같이 입력한다.
 - ssh agent 실행 여부 확인

	 - 아래 명령어를 입력하면 Agent pid xxx 출력하면 성공
```
eval "$(ssh-agent -s)"
```
 - agent에 비밀키 등록

	 - Identity added :  ~~ 메세지가 나오면 성공
```
ssh-add "Local repository 위치/My_home"
```
 - github ssh 연결 테스트

	 - You've successfully authenticated, ~~ 메세지가 나오면 성공
```
ssh -T git@github.com
```

4. ssh 접속 방법을 사용하여 github 저장소에 접근할 수 있도록 git 설정을 변경해준다.
- ssh 접속 주소 복사
<p align="center">  
<img src="https://user-images.githubusercontent.com/85601490/122719278-f152b480-d2a8-11eb-9289-6ff0ca009469.png" width="400"/>  
</p> 

```
git remote add origin 복사한 주소
```
- git remote -v 명령어로 기존에 repository를 연결했었는지 확인한 후 제거하고 다시 연결
-  상황에 따라 git pull 또는 git clone으로 github에 있는 파일을 가져온다.
```
git pull origin master
```
```
git clone ssh주소
```

5. 만약 pull/push 과정에서 Permission denied 에러가 뜬다면 config file에 정보 추가해야함!  
   repository 폴더로 이동해서 아래와 같이 해당 repository 이름과 이메일 입력하면 정상 작동 될 것!
```
git config --local user.name "github name"
git config --local user.email "github email주소"
```

6. 끝!





참고자료 : https://goddaehee.tistory.com/254
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTUzNDE4MjIyMywtNzM1MjYwMzU5LC0xNT
g2NTUwNTc4LC0yMzI4NDMxNzYsNTU2NzI0NjcxLDQ5NzgxODgx
MF19
-->
